"""
Записывает результат работы Инспектора (баллы и комментарии) в реальный
xlsx-файл. Работает только с ячейками — никогда не трогает формулы шаблона,
они пересчитаются сами при открытии файла в Excel/LibreOffice.

Колонки (см. inspector_agent_v2.md): A — шкала (не трогаем), I — оценка
инспектора, K — комментарий инспектора.
"""

import openpyxl

SCORE_COLUMN = 9  # I
COMMENT_COLUMN = 11  # K

# Ключевые слова для поиска нужной строки шапки по тексту подписи.
# Строки шапки не имеют фиксированных номеров — структура отличается
# между разными версиями шаблона (напр. "Афимолл" vs "Мясницкая"),
# поэтому ищем строку по содержанию, а не по жёсткому номеру.
HEADER_LABEL_KEYWORDS = {
    "restaurant": ["название ресторана"],
    "date": ["дата и вид инспекции", "дата и время инспекции"],
    "time_in": ["время захода"],
    "time_out": ["время выхода"],
    "waiter_name": ["имя и описание официанта", "имя официанта"],
    "waiter_description": ["имя и фамилия сотрудника с бейджа"],
}


def _find_header_row(ws, keywords: list[str], max_row: int = 15) -> int | None:
    """Ищет в первых max_row строках столбца C строку, чей текст
    содержит один из keywords (без учёта регистра)."""
    for row in range(1, max_row + 1):
        value = ws.cell(row=row, column=3).value
        if not value:
            continue
        lowered = str(value).lower()
        if any(keyword in lowered for keyword in keywords):
            return row
    return None


def apply_fill_results(
    empty_template_path: str,
    output_path: str,
    header: dict,
    filled_rows: list[dict],
) -> list[int]:
    """Открывает пустой шаблон, проставляет баллы/комментарии/шапку,
    сохраняет как новый файл (не трогая исходный).

    Возвращает список номеров строк, которые были пропущены из-за
    отсутствия комментария (защита от 'осиротевших' баллов без пояснения)."""
    wb = openpyxl.load_workbook(empty_template_path, data_only=False)
    ws = wb["шаблон"]

    for key, keywords in HEADER_LABEL_KEYWORDS.items():
        value = header.get(key)
        if not value:
            continue

        row = _find_header_row(ws, keywords)
        if row is None:
            continue  # в этой версии шаблона такого поля шапки просто нет

        cell = ws.cell(row=row, column=3)
        existing_text = cell.value or ""
        if ":" in existing_text:
            label = existing_text.split(":", 1)[0]
            cell.value = f"{label}: {value}"
        else:
            cell.value = f"{existing_text} {value}".strip()

    skipped_rows = []
    for item in filled_rows:
        row = item["row"]
        score = item["score"]
        comment = item.get("comment", "").strip()

        if not comment:
            # Без комментария балл не записываем вообще — оставляем строку
            # пустой, а не создаём "осиротевший" балл без пояснения. Это
            # частая ошибка модели: она иногда заполняет две строки одного
            # вопроса, и одна из них остаётся без текста.
            skipped_rows.append(row)
            continue

        ws.cell(row=row, column=SCORE_COLUMN, value=score)
        ws.cell(row=row, column=COMMENT_COLUMN, value=comment)

    wb.save(output_path)
    return skipped_rows


def check_for_formula_errors(output_path: str) -> list[str]:
    """Проверяет сохранённый файл на явные признаки сломанных формул.
    Полноценный пересчёт формул openpyxl не делает (это не Excel),
    поэтому здесь только базовая проверка на артефакты вроде #REF!."""
    wb = openpyxl.load_workbook(output_path, data_only=False)
    ws = wb["шаблон"]
    problems = []

    for row in ws.iter_rows():
        for cell in row:
            if isinstance(cell.value, str) and any(
                marker in cell.value for marker in ("#REF!", "#DIV/0!", "#VALUE!")
            ):
                problems.append(f"{cell.coordinate}: {cell.value}")

    return problems
