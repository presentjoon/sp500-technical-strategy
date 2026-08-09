"""`docs/daily_workflow.md`를 Notion 페이지로 밀어 넣는다.

    python scripts/push_notion.py --dry-run     # API 호출 없이 변환 결과만 확인
    python scripts/push_notion.py --list        # 부모 페이지의 자식 목록 보기
    python scripts/push_notion.py               # 실제 갱신

준비 (최초 1회)
---------------
1. https://www.notion.so/my-integrations 에서 **New integration** 생성 → 토큰 복사
2. Notion에서 대상 페이지 → **⋯ → 연결(Connections) → 방금 만든 통합 추가**
   이 단계를 빼먹으면 API가 404를 돌려준다. "페이지가 없다"가 아니라
   "통합에게 안 보인다"는 뜻이다.
3. 프로젝트 루트에 `.env` 파일을 만들고 아래 한 줄:

       NOTION_TOKEN=ntn_여기에붙여넣기

   `.env`는 `.gitignore`에 있다. **토큰을 커밋하지 않는다.**

왜 이어 붙이지 않고 덮어쓰는가
------------------------------
매일 실행하는 스크립트가 append 방식이면 같은 내용이 계속 쌓인다. 그래서
대상 페이지의 **기존 블록을 전부 지우고 새로 채운다.** 지우는 동작이 들어가므로
**전용 하위 페이지 하나만** 대상으로 삼는다. 다른 메모가 섞인 페이지를 대상으로
하면 그 메모까지 사라진다.

의존성
------
표준 라이브러리만 쓴다 (`urllib.request`). requirements.txt에 추가되는 것 없음.
"""

import argparse
import io
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()     # -> Path
PROJECT_ROOT = SCRIPT_PATH.parent.parent   # -> Path

sys.path.insert(0, str(SCRIPT_PATH.parent))

from export_notion import split_sections, strip_relative_links  # noqa: E402

SOURCE = PROJECT_ROOT / "docs" / "daily_workflow.md"  # -> Path
ENV_FILE = PROJECT_ROOT / ".env"                       # -> Path

# Road to Columbia 페이지. URL 끝의 32자리 16진수가 페이지 ID다.
# https://www.notion.so/Road-to-Columbia-3b772ed24ec2808a93a6da50dea2fe58
ROOT_PAGE_ID = "3b772ed24ec2808a93a6da50dea2fe58"  # -> str

PARENT_PAGE_TITLE = "PROJECT"          # -> str, 이 이름의 하위 페이지를 부모로 삼는다
TARGET_PAGE_TITLE = "일일 워크플로우"    # -> str, 이 이름의 페이지를 만들고 매번 덮어쓴다

NOTION_API = "https://api.notion.com/v1"  # -> str
NOTION_VERSION = "2022-06-28"             # -> str, 명시하지 않으면 API가 거부한다

BATCH_SIZE = 100  # -> int, Notion이 한 번에 받는 자식 블록 수 상한
TEXT_LIMIT = 2000  # -> int, rich_text 하나의 문자 수 상한


# ---------------------------------------------------------------------------
# 1. 토큰
# ---------------------------------------------------------------------------
def load_token():
    """환경변수 → `.env` 순으로 토큰을 찾는다."""
    token = os.environ.get("NOTION_TOKEN")  # -> str | None

    if token:
        return token.strip()

    if not ENV_FILE.exists():
        return None

    lines = io.open(ENV_FILE, encoding="utf-8").read().split("\n")  # -> list[str]

    for line in lines:
        stripped = line.strip()  # -> str

        if not stripped or stripped.startswith("#"):
            continue

        if "=" not in stripped:
            continue

        name, _, value = stripped.partition("=")  # -> (str, str, str)

        if name.strip() == "NOTION_TOKEN":
            return value.strip().strip('"').strip("'")

    return None


# ---------------------------------------------------------------------------
# 2. HTTP
# ---------------------------------------------------------------------------
def call_api(method, path, token, payload=None):
    """Notion API 호출. 실패하면 응답 본문을 그대로 보여준다.

    조용히 실패하면 "왜 안 됐는지"를 알 수 없다. Notion은 오류 메시지가
    구체적인 편이라(권한/ID/스키마) 그대로 출력하는 편이 낫다.
    """
    url = f"{NOTION_API}{path}"  # -> str

    data = None  # -> bytes | None

    if payload is not None:
        encoded = json.dumps(payload, ensure_ascii=False)  # -> str
        data = encoded.encode("utf-8")                     # -> bytes

    request = urllib.request.Request(url, data=data, method=method)  # -> Request
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Notion-Version", NOTION_VERSION)
    request.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")  # -> str
            return json.loads(body)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")  # -> str
        print(f"\n[API 오류] {method} {path} → HTTP {error.code}")
        print(detail)

        if error.code == 404:
            print(
                "\n404는 대개 '페이지가 없다'가 아니라 "
                "'통합에 페이지가 공유되지 않았다'는 뜻이다.\n"
                "Notion에서 대상 페이지 → ⋯ → 연결 → 통합을 추가했는지 확인할 것."
            )

        raise


# ---------------------------------------------------------------------------
# 3. 마크다운 인라인 → Notion rich_text
# ---------------------------------------------------------------------------
INLINE_PATTERN = re.compile(
    r"(\*\*[^*]+\*\*)"      # 굵게
    r"|(`[^`]+`)"           # 인라인 코드
    r"|(\[[^\]]+\]\([^)]+\))"  # 링크
)  # -> Pattern


def make_rich_text(text):
    """`**굵게**`, `` `코드` ``, `[표시](주소)` 세 가지만 처리한다.

    나머지 표기(기울임, LaTeX 등)는 **일반 텍스트로 남긴다.** 원본 문서에서
    수식은 드물고, 잘못 변환해서 깨뜨리는 것보다 그대로 두는 편이 낫다.
    """
    if not text:
        return []

    pieces = []      # -> list[dict]
    position = 0     # -> int

    for match in INLINE_PATTERN.finditer(text):
        if match.start() > position:
            plain = text[position:match.start()]  # -> str
            pieces.append(_text_piece(plain))

        token = match.group(0)  # -> str

        if token.startswith("**"):
            inner = token[2:-2]                                  # -> str
            pieces.append(_text_piece(inner, bold=True))
        elif token.startswith("`"):
            inner = token[1:-1]                                  # -> str
            pieces.append(_text_piece(inner, code=True))
        else:
            label, _, rest = token[1:].partition("](")           # -> (str, str, str)
            href = rest[:-1]                                     # -> str
            pieces.append(_text_piece(label, href=href))

        position = match.end()

    if position < len(text):
        pieces.append(_text_piece(text[position:]))

    trimmed = [piece for piece in pieces if piece["text"]["content"]]  # -> list[dict]

    return trimmed[:100]  # rich_text 배열 상한


def _text_piece(content, bold=False, code=False, href=None):
    """rich_text 원소 하나."""
    clipped = content[:TEXT_LIMIT]  # -> str

    piece = {
        "type": "text",
        "text": {"content": clipped},
        "annotations": {"bold": bold, "code": code},
    }  # -> dict

    if href:
        piece["text"]["link"] = {"url": href}

    return piece


# ---------------------------------------------------------------------------
# 4. 마크다운 블록 → Notion 블록
# ---------------------------------------------------------------------------
def markdown_to_blocks(markdown):
    """마크다운 본문을 Notion 블록 리스트로 바꾼다.

    처리 대상: 제목(h1~h4), 문단, 인용, 글머리표, 번호목록, 표, 코드펜스, 구분선.
    이미지는 로컬 경로라 Notion에서 열 수 없으므로 **안내 문단으로 바꾼다.**
    """
    lines = markdown.split("\n")  # -> list[str]

    blocks = []   # -> list[dict]
    index = 0     # -> int

    while index < len(lines):
        line = lines[index]      # -> str
        stripped = line.strip()  # -> str

        if not stripped:
            index = index + 1
            continue

        # --- 코드 펜스 ---
        if stripped.startswith("```"):
            index, block = _read_code_fence(lines, index)
            blocks.append(block)
            continue

        # --- 표 ---
        if stripped.startswith("|"):
            index, block = _read_table(lines, index)

            if block is not None:
                blocks.append(block)

            continue

        # --- 인용 (연속 줄을 하나로 묶는다) ---
        if stripped.startswith("> ") or stripped == ">":
            index, block = _read_quote(lines, index)
            blocks.append(block)
            continue

        # --- 구분선 ---
        if stripped == "---":
            blocks.append({"object": "block", "type": "divider", "divider": {}})
            index = index + 1
            continue

        # --- 제목 ---
        heading = _read_heading(stripped)  # -> dict | None

        if heading is not None:
            blocks.append(heading)
            index = index + 1
            continue

        # --- 이미지 ---
        if stripped.startswith("!["):
            label = re.sub(r"!\[([^\]]*)\].*", r"\1", stripped)  # -> str
            note = f"🖼️ 그림: {label} — 레포 figures/ 참조"        # -> str
            blocks.append(_paragraph(note))
            index = index + 1
            continue

        # --- 글머리표 / 번호목록 ---
        list_block = _read_list_item(stripped)  # -> dict | None

        if list_block is not None:
            blocks.append(list_block)
            index = index + 1
            continue

        # --- 문단 (줄바꿈으로 이어진 여러 줄을 한 문단으로) ---
        index, block = _read_paragraph(lines, index)
        blocks.append(block)

    return blocks


def _read_heading(stripped):
    """`#`~`####`를 Notion 제목 블록으로. Notion에는 h3까지만 있다."""
    match = re.match(r"^(#{1,6})\s+(.*)$", stripped)  # -> Match | None

    if match is None:
        return None

    level = len(match.group(1))  # -> int
    body = match.group(2)        # -> str

    capped = min(level, 3)       # -> int, h4 이하는 전부 h3으로
    key = f"heading_{capped}"    # -> str

    block = {
        "object": "block",
        "type": key,
        key: {"rich_text": make_rich_text(body), "is_toggleable": False},
    }  # -> dict

    return block


def _read_list_item(stripped):
    """`- ` / `* ` / `1. `를 목록 블록으로."""
    bullet = re.match(r"^[-*]\s+(.*)$", stripped)  # -> Match | None

    if bullet is not None:
        return {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": make_rich_text(bullet.group(1))},
        }

    numbered = re.match(r"^\d+\.\s+(.*)$", stripped)  # -> Match | None

    if numbered is not None:
        return {
            "object": "block",
            "type": "numbered_list_item",
            "numbered_list_item": {"rich_text": make_rich_text(numbered.group(1))},
        }

    return None


def _read_quote(lines, index):
    """연속된 `>` 줄을 인용 블록 하나로 묶는다."""
    collected = []  # -> list[str]

    while index < len(lines):
        stripped = lines[index].strip()  # -> str

        if not (stripped.startswith("> ") or stripped == ">"):
            break

        collected.append(stripped[2:] if stripped.startswith("> ") else "")
        index = index + 1

    joined = " ".join(part for part in collected if part)  # -> str

    block = {
        "object": "block",
        "type": "quote",
        "quote": {"rich_text": make_rich_text(joined)},
    }  # -> dict

    return index, block


def _read_paragraph(lines, index):
    """빈 줄이 나올 때까지를 한 문단으로 묶는다."""
    collected = []  # -> list[str]

    while index < len(lines):
        stripped = lines[index].strip()  # -> str

        if not stripped:
            break

        starts_other_block = (
            stripped.startswith("|")
            or stripped.startswith("> ")
            or stripped.startswith("```")
            or stripped.startswith("#")
            or stripped == "---"
        )  # -> bool

        if starts_other_block and collected:
            break

        collected.append(stripped)
        index = index + 1

    joined = " ".join(collected)  # -> str

    return index, _paragraph(joined)


def _paragraph(text):
    block = {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": make_rich_text(text)},
    }  # -> dict

    return block


def _read_code_fence(lines, index):
    """```로 열고 닫는 구간을 코드 블록으로."""
    index = index + 1  # 여는 줄 건너뛰기
    collected = []     # -> list[str]

    while index < len(lines):
        if lines[index].strip().startswith("```"):
            index = index + 1
            break

        collected.append(lines[index])
        index = index + 1

    content = "\n".join(collected)[:TEXT_LIMIT]  # -> str

    block = {
        "object": "block",
        "type": "code",
        "code": {
            "rich_text": [{"type": "text", "text": {"content": content}}],
            "language": "plain text",
        },
    }  # -> dict

    return index, block


def _read_table(lines, index):
    """`|`로 시작하는 연속 줄을 표 블록으로.

    두 번째 줄(`|---|---|`)은 구분선이므로 행으로 세지 않는다.
    """
    rows = []  # -> list[list[str]]

    while index < len(lines):
        stripped = lines[index].strip()  # -> str

        if not stripped.startswith("|"):
            break

        is_separator = re.match(r"^\|[\s:\-|]+\|$", stripped) is not None  # -> bool

        if not is_separator:
            inner = stripped.strip("|")                       # -> str
            cells = [cell.strip() for cell in inner.split("|")]  # -> list[str]
            rows.append(cells)

        index = index + 1

    if not rows:
        return index, None

    width = max(len(row) for row in rows)  # -> int

    table_rows = []  # -> list[dict]

    for row in rows:
        padded = row + [""] * (width - len(row))  # -> list[str]
        cells = [make_rich_text(cell) for cell in padded]  # -> list[list[dict]]

        table_rows.append({
            "object": "block",
            "type": "table_row",
            "table_row": {"cells": cells},
        })

    block = {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": width,
            "has_column_header": True,
            "has_row_header": False,
            "children": table_rows,
        },
    }  # -> dict

    return index, block


# ---------------------------------------------------------------------------
# 5. 문서 전체 → 블록 (날짜 절은 토글로)
# ---------------------------------------------------------------------------
def build_document_blocks(markdown):
    """날짜 절은 토글, 나머지는 펼친 상태로 만든다.

    Returns
    -------
    (list[dict], list[tuple[int, list[dict]]])
        상위 블록 목록과, [(상위 블록 인덱스, 그 토글의 자식 블록), ...]

    왜 토글 자식을 분리해서 돌려주는가
    ----------------------------------
    Notion API는 한 번의 요청에서 **2단계 중첩까지만** 받는다. 토글 안에 표가
    들어가면 토글 → 표 → 행으로 3단계가 되어 거부된다. 그래서 토글은 **빈 채로
    먼저 만들고**, 자식은 그 토글을 부모로 삼아 두 번째 요청으로 넣는다.
    """
    cleaned = strip_relative_links(markdown)  # -> str

    preamble, sections = split_sections(cleaned)  # -> (str, list[tuple])

    preamble_lines = preamble.split("\n")  # -> list[str]
    body_lines = [line for line in preamble_lines[1:]]  # 첫 줄(제목)은 페이지 제목이 된다

    top_blocks = []      # -> list[dict]
    pending_children = []  # -> list[tuple[int, list[dict]]]

    banner = (
        "⚠️ 이 페이지는 사본입니다. 정본은 GitHub 레포의 `docs/daily_workflow.md`입니다. "
        "여기서 직접 고치지 마세요 — 다음 갱신 때 덮어써집니다. "
        "모든 **수치의 정본**은 레포의 `reports/week2_key_numbers.md`입니다."
    )  # -> str

    top_blocks.append({
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": make_rich_text(banner),
            "icon": {"type": "emoji", "emoji": "⚠️"},
        },
    })

    top_blocks.extend(markdown_to_blocks("\n".join(body_lines)))

    day_pattern = re.compile(r"^D\d")  # -> Pattern

    for title, body in sections:
        is_day = bool(day_pattern.match(title))  # -> bool

        if is_day:
            toggle = {
                "object": "block",
                "type": "toggle",
                "toggle": {"rich_text": make_rich_text(f"**{title}**")},
            }  # -> dict

            top_blocks.append(toggle)
            pending_children.append((len(top_blocks) - 1, markdown_to_blocks(body)))
        else:
            top_blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": make_rich_text(title), "is_toggleable": False},
            })
            top_blocks.extend(markdown_to_blocks(body))

    return top_blocks, pending_children


# ---------------------------------------------------------------------------
# 6. 페이지 찾기 / 만들기 / 비우기
# ---------------------------------------------------------------------------
def list_children(page_id, token):
    """자식 블록 전체 (페이지네이션 따라감)."""
    results = []      # -> list[dict]
    cursor = None     # -> str | None

    while True:
        path = f"/blocks/{page_id}/children?page_size=100"  # -> str

        if cursor:
            path = f"{path}&start_cursor={cursor}"

        response = call_api("GET", path, token)  # -> dict
        results.extend(response.get("results", []))

        if not response.get("has_more"):
            break

        cursor = response.get("next_cursor")

    return results


def child_page_title(block):
    """`child_page` 블록의 제목."""
    if block.get("type") != "child_page":
        return None

    return block["child_page"].get("title")


def find_child_page(parent_id, title, token):
    """이름이 일치하는 하위 페이지 ID. 없으면 None."""
    children = list_children(parent_id, token)  # -> list[dict]

    for block in children:
        if child_page_title(block) == title:
            return block["id"]

    return None


def create_child_page(parent_id, title, token):
    """빈 하위 페이지를 만든다."""
    payload = {
        "parent": {"type": "page_id", "page_id": parent_id},
        "properties": {
            "title": [{"type": "text", "text": {"content": title}}]
        },
    }  # -> dict

    page = call_api("POST", "/pages", token, payload)  # -> dict

    return page["id"]


def clear_page(page_id, token):
    """페이지의 기존 블록을 전부 삭제한다.

    Notion의 블록 삭제는 **휴지통으로 보내는 것**이라 되돌릴 수 있다.
    그래도 대상은 전용 페이지 하나로 제한한다.
    """
    children = list_children(page_id, token)  # -> list[dict]

    for block in children:
        call_api("DELETE", f"/blocks/{block['id']}", token)

    return len(children)


def append_blocks(parent_id, blocks, token):
    """자식 블록을 100개씩 나눠 붙인다.

    Returns
    -------
    list[dict]
        붙인 블록들의 응답 (id를 얻기 위해 필요)
    """
    appended = []  # -> list[dict]

    position = 0  # -> int

    while position < len(blocks):
        chunk = blocks[position:position + BATCH_SIZE]  # -> list[dict]

        response = call_api(
            "PATCH",
            f"/blocks/{parent_id}/children",
            token,
            {"children": chunk},
        )  # -> dict

        appended.extend(response.get("results", []))
        position = position + BATCH_SIZE

    return appended


# ---------------------------------------------------------------------------
# 7. 엔트리포인트
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="daily_workflow.md를 Notion에 갱신한다")
    parser.add_argument("--dry-run", action="store_true",
                        help="API를 호출하지 않고 변환 결과만 요약한다")
    parser.add_argument("--list", action="store_true",
                        help="부모 페이지의 자식 목록을 출력하고 끝낸다")
    parser.add_argument("--parent", default=None,
                        help="부모 페이지 ID를 직접 지정한다 (PROJECT를 못 찾을 때)")
    parser.add_argument("--dump", default=None,
                        help="변환된 블록 JSON을 이 경로에 저장한다 (점검용)")

    args = parser.parse_args()  # -> Namespace

    if not SOURCE.exists():
        print(f"원본이 없다: {SOURCE}")
        return 1

    markdown = io.open(SOURCE, encoding="utf-8").read()  # -> str

    top_blocks, pending_children = build_document_blocks(markdown)  # -> (list, list)

    child_total = sum(len(children) for _, children in pending_children)  # -> int

    print(f"원본: {SOURCE.name}")
    print(f"  상위 블록 {len(top_blocks)}개")
    print(f"  토글 {len(pending_children)}개 · 토글 내부 블록 {child_total}개")

    if args.dump:
        payload = {"top": top_blocks, "children": pending_children}  # -> dict
        io.open(args.dump, "w", encoding="utf-8").write(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )
        print(f"  블록 JSON 저장: {args.dump}")

    if args.dry_run:
        print("\n--dry-run 이므로 여기서 멈춘다. API 호출 없음.")
        _print_block_summary(top_blocks)
        return 0

    token = load_token()  # -> str | None

    if not token:
        print(
            "\nNOTION_TOKEN을 찾을 수 없다.\n"
            f"  환경변수로 넣거나, {ENV_FILE} 에 아래 한 줄을 적을 것:\n"
            "      NOTION_TOKEN=ntn_..."
        )
        return 1

    if args.list:
        parent_id = args.parent or ROOT_PAGE_ID  # -> str
        children = list_children(parent_id, token)  # -> list[dict]

        print(f"\n{parent_id} 의 자식 {len(children)}개:")

        for block in children:
            title = child_page_title(block)  # -> str | None
            label = f"[하위 페이지] {title}" if title else f"[{block['type']}]"
            print(f"  {block['id']}  {label}")

        return 0

    # --- 부모 찾기 ---
    if args.parent:
        parent_id = args.parent  # -> str
        print(f"\n부모(지정): {parent_id}")
    else:
        parent_id = find_child_page(ROOT_PAGE_ID, PARENT_PAGE_TITLE, token)  # -> str | None

        if parent_id is None:
            print(
                f"\n'{PARENT_PAGE_TITLE}' 하위 페이지를 찾지 못했다.\n"
                "  --list 로 실제 자식 목록을 확인하고, --parent <id> 로 지정할 것.\n"
                "  PROJECT가 하위 페이지가 아니라 본문 속 제목이면 하위 페이지로 만들어야 한다."
            )
            return 1

        print(f"\n부모: {PARENT_PAGE_TITLE} ({parent_id})")

    # --- 대상 페이지 찾기 또는 생성 ---
    target_id = find_child_page(parent_id, TARGET_PAGE_TITLE, token)  # -> str | None

    if target_id is None:
        target_id = create_child_page(parent_id, TARGET_PAGE_TITLE, token)  # -> str
        print(f"대상 페이지 생성: {TARGET_PAGE_TITLE} ({target_id})")
    else:
        removed = clear_page(target_id, token)  # -> int
        print(f"대상 페이지: {TARGET_PAGE_TITLE} ({target_id}) — 기존 블록 {removed}개 삭제")

    # --- 상위 블록 ---
    appended = append_blocks(target_id, top_blocks, token)  # -> list[dict]
    print(f"상위 블록 {len(appended)}개 추가")

    # --- 토글 자식 ---
    for block_index, children in pending_children:
        if not children:
            continue

        toggle_id = appended[block_index]["id"]  # -> str
        append_blocks(toggle_id, children, token)

    print(f"토글 자식 {child_total}개 추가")
    print(f"\n완료: https://www.notion.so/{target_id.replace('-', '')}")

    return 0


def _print_block_summary(blocks):
    """블록 타입별 개수. 변환이 의도대로 됐는지 눈으로 확인하는 용도."""
    counts = {}  # -> dict[str, int]

    for block in blocks:
        block_type = block["type"]                       # -> str
        counts[block_type] = counts.get(block_type, 0) + 1

    print("\n상위 블록 타입별 개수:")

    for block_type in sorted(counts):
        print(f"  {block_type:24s} {counts[block_type]}")


if __name__ == "__main__":
    sys.exit(main())
