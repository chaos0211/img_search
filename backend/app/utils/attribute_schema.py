from __future__ import annotations

from dataclasses import dataclass


LABEL_ORDER = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

LABEL_TEXT = {
    "airplane": "airplane",
    "automobile": "automobile",
    "bird": "bird",
    "cat": "cat",
    "deer": "deer",
    "dog": "dog",
    "frog": "frog",
    "horse": "horse",
    "ship": "ship",
    "truck": "truck",
}

SUPER_CATEGORY = {
    "airplane": ("vehicle", "交通工具"),
    "automobile": ("vehicle", "交通工具"),
    "ship": ("vehicle", "交通工具"),
    "truck": ("vehicle", "交通工具"),
    "bird": ("animal", "动物"),
    "cat": ("animal", "动物"),
    "deer": ("animal", "动物"),
    "dog": ("animal", "动物"),
    "frog": ("animal", "动物"),
    "horse": ("animal", "动物"),
}

OBJECT_TYPE = {
    "airplane": ("aircraft", "飞行器"),
    "automobile": ("land_vehicle", "陆地车辆"),
    "truck": ("land_vehicle", "陆地车辆"),
    "ship": ("vessel", "船舶"),
    "cat": ("mammal", "哺乳动物"),
    "deer": ("mammal", "哺乳动物"),
    "dog": ("mammal", "哺乳动物"),
    "horse": ("mammal", "哺乳动物"),
    "bird": ("bird", "鸟类"),
    "frog": ("amphibian", "两栖类"),
}

CLUSTER_TAG = {
    "airplane": ("0", "cluster 0"),
    "automobile": ("0", "cluster 0"),
    "ship": ("0", "cluster 0"),
    "truck": ("0", "cluster 0"),
    "cat": ("1", "cluster 1"),
    "deer": ("1", "cluster 1"),
    "dog": ("1", "cluster 1"),
    "horse": ("1", "cluster 1"),
    "bird": ("2", "cluster 2"),
    "frog": ("2", "cluster 2"),
}

OPTION_ORDER = {
    "super": ["vehicle", "animal"],
    "object": ["aircraft", "land_vehicle", "vessel", "mammal", "bird", "amphibian"],
    "cluster": ["0", "1", "2"],
}


@dataclass(frozen=True)
class ImageAttributes:
    basic_category: str
    basic_category_label: str
    super_category: str
    super_category_label: str
    object_type: str
    object_type_label: str
    cluster_tag: str
    cluster_tag_label: str
    attribute_values: list[str]
    display_tags: list[dict[str, str]]


def image_attributes(label_name: str | None) -> ImageAttributes:
    label = str(label_name or "").strip()
    super_value, super_label = SUPER_CATEGORY.get(label, ("unknown", "未知"))
    object_value, object_label = OBJECT_TYPE.get(label, ("unknown", "未知"))
    cluster_value, cluster_label = CLUSTER_TAG.get(label, ("unknown", "unknown"))
    basic_label = LABEL_TEXT.get(label, label or "unknown")
    values = [
        f"basic:{label}",
        f"super:{super_value}",
        f"object:{object_value}",
        f"cluster:{cluster_value}",
    ]
    return ImageAttributes(
        basic_category=label,
        basic_category_label=basic_label,
        super_category=super_value,
        super_category_label=super_label,
        object_type=object_value,
        object_type_label=object_label,
        cluster_tag=cluster_value,
        cluster_tag_label=cluster_label,
        attribute_values=values,
        display_tags=[
            {"label": basic_label, "value": f"basic:{label}"},
            {"label": super_label, "value": f"super:{super_value}"},
            {"label": object_label, "value": f"object:{object_value}"},
            {"label": cluster_label, "value": f"cluster:{cluster_value}"},
        ],
    )


def attribute_options(counts: dict[str, int]) -> list[dict]:
    return [
        {
            "key": "basic",
            "label": "基础类别",
            "options": [
                {"value": f"basic:{label}", "label": label, "count": int(counts.get(label, 0))}
                for label in LABEL_ORDER
                if label in counts
            ],
        },
        {
            "key": "super",
            "label": "大类",
            "options": _group_options(counts, "super", SUPER_CATEGORY),
        },
        {
            "key": "object",
            "label": "对象类型",
            "options": _group_options(counts, "object", OBJECT_TYPE),
        },
        {
            "key": "cluster",
            "label": "聚类标签",
            "options": _group_options(counts, "cluster", CLUSTER_TAG),
        },
    ]


def selected_attribute_labels(values: list[str]) -> list[dict[str, str]]:
    option_lookup = {
        option["value"]: option["label"]
        for group in attribute_options({label: 1 for label in LABEL_ORDER})
        for option in group["options"]
    }
    labels = []
    seen = set()
    for value in values:
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        labels.append({"value": value, "label": option_lookup.get(value, value)})
    return labels


def matches_attribute_values(item_values: list[str] | tuple[str, ...] | None, selected_values: set[str]) -> bool:
    if not selected_values:
        return True
    return selected_values.issubset(set(item_values or []))


def _group_options(counts: dict[str, int], prefix: str, mapping: dict[str, tuple[str, str]]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for label, total in counts.items():
        value, display_label = mapping.get(label, ("unknown", "未知"))
        if value == "unknown":
            continue
        key = f"{prefix}:{value}"
        item = grouped.setdefault(key, {"value": key, "label": display_label, "count": 0})
        item["count"] += int(total)
    order = OPTION_ORDER.get(prefix, [])
    return sorted(grouped.values(), key=lambda item: order.index(item["value"].split(":", 1)[1]) if item["value"].split(":", 1)[1] in order else len(order))
