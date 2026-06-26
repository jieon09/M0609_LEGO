import json
from typing import Union


def get_brick_info(json_input: Union[str, dict], index_input: int) -> dict:
    """인덱스에 해당하는 블록 정보를 반환.

    Args:
        json_input: bricks 배열을 포함한 JSON 문자열 또는 dict
        index_input: 조회할 블록 인덱스 (음수 인덱스 허용)

    Returns:
        {"total_count": int, "target_brick_type": str, "target_brick_color": str}
        범위 초과 시 {"error": str}
    """
    data = json.loads(json_input) if isinstance(json_input, str) else json_input
    bricks_list = data.get("bricks", [])
    total_count = len(bricks_list)

    if index_input >= total_count or index_input < -total_count:
        return {
            "error": f"입력한 인덱스 {index_input}은 범위를 벗어났습니다. (유효 범위: {-total_count} ~ {total_count - 1})"
        }

    target_brick = bricks_list[index_input]
    return {
        "total_count": total_count,
        "target_brick_type": f"{target_brick['width']}x{target_brick['height']}",
        "target_brick_color": target_brick['color'],
    }


if __name__ == "__main__":
    _test_data = """
    {
      "brick_count": 17,
      "bricks": [
        {"row": 0, "col": 22, "width": 2, "height": 2, "color": "B"},
        {"row": 1, "col": 18, "width": 3, "height": 2, "color": "B"},
        {"row": 2, "col": 9, "width": 3, "height": 2, "color": "B"},
        {"row": 2, "col": 14, "width": 2, "height": 2, "color": "B"},
        {"row": 3, "col": 5, "width": 3, "height": 2, "color": "B"},
        {"row": 4, "col": 3, "width": 2, "height": 2, "color": "B"},
        {"row": 4, "col": 12, "width": 2, "height": 2, "color": "B"},
        {"row": 7, "col": 0, "width": 2, "height": 2, "color": "B"},
        {"row": 7, "col": 4, "width": 2, "height": 2, "color": "B"},
        {"row": 10, "col": 4, "width": 2, "height": 3, "color": "B"},
        {"row": 13, "col": 2, "width": 2, "height": 3, "color": "B"},
        {"row": 13, "col": 4, "width": 2, "height": 2, "color": "B"},
        {"row": 14, "col": 0, "width": 2, "height": 2, "color": "B"},
        {"row": 16, "col": 3, "width": 2, "height": 3, "color": "B"},
        {"row": 19, "col": 2, "width": 2, "height": 3, "color": "B"},
        {"row": 19, "col": 6, "width": 3, "height": 2, "color": "B"},
        {"row": 22, "col": 3, "width": 2, "height": 2, "color": "B"}
      ]
    }
    """
    print("--- 테스트 1 (인덱스 -1) ---")
    print(json.dumps(get_brick_info(_test_data, -1), indent=2, ensure_ascii=False))
    print("\n--- 테스트 2 (인덱스 1) ---")
    print(json.dumps(get_brick_info(_test_data, 1), indent=2, ensure_ascii=False))
