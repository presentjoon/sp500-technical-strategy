"""`docs/daily_workflow.md`를 Notion에 붙여넣을 형태로 변환한다.

    python scripts/export_notion.py

왜 손으로 쓰지 않고 변환하는가
------------------------------
Notion 페이지는 본질적으로 **레포 문서의 복사본**이다. 손으로 옮겨 적으면
D9에서 겪은 것과 같은 일이 반복된다 — 같은 내용이 두 곳에 있으면 언젠가 갈린다.
그래서 Notion용 파일은 **직접 편집하지 않고 매번 이 스크립트로 다시 만든다.**
정본은 언제나 `docs/daily_workflow.md`이고, Notion은 읽기용 사본이다.

무엇을 바꾸는가
---------------
1. 날짜별 `## D{n}` 절을 **토글**(`<details>`/`<summary>`)로 감싼다.
   Notion에서 토글은 접히므로, 9개 날짜가 한 화면에 목차처럼 보인다.
2. 절 안의 `###` 소제목은 한 단계 내려 `####`로 만든다.
   토글 안에서 `###`는 너무 크게 잡힌다.
3. 레포 상대 경로 링크(`../reports/...`)는 Notion에서 깨지므로 제거하고
   **코드 표기만 남긴다.**
4. 맨 위에 "정본은 GitHub" 배너를 붙인다. 이 배너가 없으면 Notion을 원본으로
   착각하게 된다.

출력: `docs/notion_project.md`
"""

import io
import re
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()   # -> Path
PROJECT_ROOT = SCRIPT_PATH.parent.parent  # -> Path

SOURCE = PROJECT_ROOT / "docs" / "daily_workflow.md"        # -> Path
TARGET = PROJECT_ROOT / "docs" / "notion_project.md"        # -> Path

REPO_URL = "https://github.com/chris-hyun/sp500-technical-strategy"  # 미설정 시 아래 배너 문구만 쓰인다

BANNER = """> ⚠️ **이 페이지는 사본입니다. 정본은 GitHub 레포의 `docs/daily_workflow.md`입니다.**
> 여기서 직접 고치지 마세요 — 다음 갱신 때 덮어써집니다.
> 수정은 레포에서 하고 `python scripts/export_notion.py`로 다시 내보냅니다.
> 모든 **수치의 정본**은 레포의 `reports/week2_key_numbers.md`입니다.

---
"""

FOOTER_TEMPLATE = """
---

_이 페이지는 `scripts/export_notion.py`가 `docs/daily_workflow.md`에서 자동 생성했습니다._
_마지막 갱신: {stamp}_
"""


def strip_relative_links(text):
    """`[표시](../경로)` 형태를 `표시`만 남긴다.

    Notion에는 레포의 상대 경로가 존재하지 않으므로 링크가 전부 깨진다.
    깨진 링크를 남기는 것보다 코드 표기만 남기는 편이 낫다.
    """
    pattern = r"\[([^\]]+)\]\((?:\.\./|\./)[^)]+\)"  # -> str
    cleaned = re.sub(pattern, r"\1", text)           # -> str

    return cleaned


def strip_local_images(text):
    """`![설명](../figures/x.png)`는 Notion에서 렌더링되지 않으므로 안내로 바꾼다."""
    pattern = r"!\[([^\]]*)\]\((?:\.\./|\./)[^)]+\)"  # -> str
    replacement = r"> 🖼️ 그림: \1 — 레포 `figures/` 참조"  # -> str
    cleaned = re.sub(pattern, replacement, text)      # -> str

    return cleaned


def demote_headings(body):
    """토글 내부의 `### `를 `#### `로 한 단계 내린다.

    코드 펜스 안의 `#`는 주석이므로 건드리면 안 된다. 그래서 펜스 상태를
    추적하면서 바깥 줄에만 적용한다.
    """
    lines = body.split("\n")  # -> list[str]
    output = []               # -> list[str]
    in_fence = False          # -> bool

    for line in lines:
        if line.startswith("```"):
            in_fence = not in_fence
            output.append(line)
            continue

        if not in_fence and line.startswith("### "):
            output.append("#" + line)
        else:
            output.append(line)

    return "\n".join(output)


def split_sections(text):
    """`## `로 시작하는 절 단위로 자른다.

    Returns
    -------
    (str, list[tuple[str, str]])
        머리말, [(제목, 본문), ...]
    """
    lines = text.split("\n")  # -> list[str]

    preamble = []   # -> list[str]
    sections = []   # -> list[tuple[str, str]]

    current_title = None  # -> str | None
    current_body = []     # -> list[str]
    in_fence = False      # -> bool

    for line in lines:
        if line.startswith("```"):
            in_fence = not in_fence

        is_section_head = (not in_fence) and line.startswith("## ")

        if is_section_head:
            if current_title is None:
                preamble = current_body
            else:
                sections.append((current_title, "\n".join(current_body)))

            current_title = line[3:].strip()  # -> str
            current_body = []
            continue

        current_body.append(line)

    if current_title is None:
        preamble = current_body
    else:
        sections.append((current_title, "\n".join(current_body)))

    return "\n".join(preamble), sections


def to_toggle(title, body):
    """한 절을 Notion 토글로 감싼다.

    `<details>` 앞뒤로 빈 줄을 둬야 마크다운 파서가 내부를 마크다운으로 읽는다.
    """
    demoted = demote_headings(body)  # -> str
    trimmed = demoted.strip("\n")    # -> str

    block = (
        "<details>\n"
        f"<summary><b>{title}</b></summary>\n"
        "\n"
        f"{trimmed}\n"
        "\n"
        "</details>\n"
    )  # -> str

    return block


def main():
    if not SOURCE.exists():
        print(f"원본이 없다: {SOURCE}")
        return 1

    raw = io.open(SOURCE, encoding="utf-8").read()  # -> str

    text = strip_local_images(raw)      # -> str
    text = strip_relative_links(text)   # -> str

    preamble, sections = split_sections(text)  # -> (str, list[tuple])

    # 머리말에서 첫 줄(제목)만 살리고 나머지는 배너로 대체한다.
    preamble_lines = preamble.split("\n")            # -> list[str]
    title_line = preamble_lines[0].strip()           # -> str, "# 날짜별 작업 기록 — ..."

    parts = [title_line, "", BANNER]  # -> list[str]

    # 날짜 절과 그 외 절을 나눈다. 날짜 절만 토글로 접는다.
    day_pattern = re.compile(r"^D\d")  # -> Pattern

    for title, body in sections:
        is_day = bool(day_pattern.match(title))  # -> bool

        if is_day:
            parts.append(to_toggle(title, body))
        else:
            parts.append(f"## {title}\n")
            parts.append(body.strip("\n"))
            parts.append("")

    stamp = _source_stamp()  # -> str
    parts.append(FOOTER_TEMPLATE.format(stamp=stamp))

    output = "\n".join(parts)  # -> str

    io.open(TARGET, "w", encoding="utf-8").write(output)

    day_count = sum(1 for title, _ in sections if day_pattern.match(title))  # -> int

    print(f"생성: {TARGET}")
    print(f"  토글로 접은 날짜 절: {day_count}개")
    print(f"  전체 절: {len(sections)}개")
    print(f"  출력 크기: {len(output.encode('utf-8')):,} bytes")

    return 0


def _source_stamp():
    """원본의 마지막 커밋 날짜를 쓴다. git이 없으면 파일 수정 시각."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ad", "--date=short", "--", str(SOURCE)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )  # -> CompletedProcess

        stamp = result.stdout.strip()  # -> str

        if stamp:
            return f"{stamp} (docs/daily_workflow.md 최종 커밋 기준)"
    except Exception:
        pass

    import datetime

    modified = SOURCE.stat().st_mtime                        # -> float
    when = datetime.datetime.fromtimestamp(modified).date()  # -> date

    return f"{when} (파일 수정 시각 기준)"


if __name__ == "__main__":
    sys.exit(main())
