"""
Mô hình dữ liệu của Knowledge Base (Phase 2).
- PlaybookCard (tầng A): "ca bệnh" triệu chứng → công thức 5 công cụ → chống chỉ định.
- Principle (tầng B): đoạn nguyên lý nhiếp ảnh / xử lý ảnh để giải thích và định hướng.
Validator bảo đảm card không bao giờ lệch khỏi toolbox (ALLOWED_OPERATIONS, PARAMETER_BOUNDS)
và từ vựng chẩn đoán (DEFECT_TYPES, SCENE_TYPES, PRESERVE_ASPECTS).
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from ..planner import ALLOWED_OPERATIONS, PARAMETER_BOUNDS
from ..state import DEFECT_TYPES, PRESERVE_ASPECTS, SCENE_TYPES

# Giá trị tham số: cố định, hoặc bảng theo severity {1: 0.5, 2: 0.8, 3: 1.0}
ParamValue = Union[float, int, str]
ParamSpec = Union[ParamValue, Dict[int, ParamValue]]

# Vùng trong điều kiện: "full" (toàn ảnh), "region" (mọi vùng khác full), "any", hoặc tên vùng
REGION_SELECTORS = ("full", "region", "any")
# Toán tử điều kiện trên chỉ số toàn cục của Module 1
METRIC_OPERATORS = ("in", "lt", "le", "gt", "ge")


class DefectCondition(BaseModel):
    """Điều kiện khớp một lỗi trong DiagnosisReport."""

    types: List[str] = Field(..., min_length=1)
    region: str = "any"
    min_severity: int = Field(default=1, ge=0, le=3)

    @field_validator("types")
    @classmethod
    def _known_types(cls, types: List[str]) -> List[str]:
        """Loại lỗi phải thuộc từ vựng chẩn đoán DEFECT_TYPES."""
        unknown = sorted(set(types) - set(DEFECT_TYPES))
        if unknown:
            raise ValueError(f"unknown defect types {unknown}")
        return types

    @field_validator("region")
    @classmethod
    def _normalize_region(cls, region: str) -> str:
        """Chuẩn hóa bộ chọn vùng về chữ thường, bỏ khoảng trắng."""
        return region.strip().casefold()


class RecipeStep(BaseModel):
    """Một bước của công thức; region '$defect' gắn với vùng của lỗi đã khớp."""

    operation: str
    region: str = "full"
    params: Dict[str, ParamSpec] = Field(default_factory=dict)
    feather_radius: Optional[int] = Field(default=None, ge=5, le=50)
    face_mode: Optional[Literal["bbox", "oval"]] = None

    @model_validator(mode="after")
    def _within_toolbox(self) -> "RecipeStep":
        """Operation thuộc toolbox; tham số đúng tên, khóa severity 0..3, trong biên."""
        if self.operation not in ALLOWED_OPERATIONS:
            raise ValueError(f"operation '{self.operation}' is not in the toolbox")
        bounds = PARAMETER_BOUNDS[self.operation]
        for name, spec in self.params.items():
            if name not in bounds:
                raise ValueError(f"{self.operation} has no parameter '{name}'")
            values = spec.values() if isinstance(spec, dict) else [spec]
            if isinstance(spec, dict) and any(key not in (0, 1, 2, 3) for key in spec):
                raise ValueError(f"{self.operation}.{name}: severity keys must be 0..3")
            for value in values:
                _check_param_value(self.operation, name, value, bounds[name])
        return self


def _check_param_value(operation: str, name: str, value: Any, constraint: Dict[str, Any]) -> None:
    """Giá trị tham số phải nằm trong PARAMETER_BOUNDS (planner sẽ không phải kẹp)."""
    if "allowed" in constraint:
        if value not in constraint["allowed"]:
            raise ValueError(f"{operation}.{name}={value!r} not in {constraint['allowed']}")
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{operation}.{name}={value!r} is not a number")
    if not constraint["min"] <= value <= constraint["max"]:
        raise ValueError(
            f"{operation}.{name}={value} outside [{constraint['min']}, {constraint['max']}]"
        )


class PlaybookCard(BaseModel):
    """Một ca bệnh trong playbook (tầng A của Knowledge Base)."""

    id: str = Field(..., pattern=r"^[a-z0-9][a-z0-9-]*$")
    title: str
    priority: int = Field(default=0, description="Cao hơn → ưu tiên khi hai card trùng thao tác")
    # True: rule engine offline được tự áp dụng; False: chỉ tư vấn cho VLM (cần ngữ cảnh)
    auto_apply: bool = False
    # Rỗng → mọi loại cảnh; có giá trị → chỉ áp dụng cho các cảnh này
    scenes: List[str] = Field(default_factory=list)
    match: Literal["any", "all"] = "any"
    defects: List[DefectCondition] = Field(default_factory=list)
    unless: List[DefectCondition] = Field(default_factory=list)
    blocked_by: List[str] = Field(default_factory=list)
    metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    recipe: List[RecipeStep] = Field(default_factory=list)
    avoid: List[str] = Field(default_factory=list)
    rationale: str = ""
    tags: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "PlaybookCard":
        """Kiểm tra từ vựng (cảnh, preserve), toán tử chỉ số và tính truy xuất được."""
        unknown_scenes = sorted(set(self.scenes) - set(SCENE_TYPES))
        if unknown_scenes:
            raise ValueError(f"unknown scenes {unknown_scenes}")
        unknown_aspects = sorted(set(self.blocked_by) - set(PRESERVE_ASPECTS))
        if unknown_aspects:
            raise ValueError(f"unknown preserve aspects {unknown_aspects}")
        if not self.defects and not self.scenes:
            raise ValueError("a card needs defect conditions or scenes to be retrievable")
        if self.auto_apply and not self.recipe:
            raise ValueError("an auto_apply card needs a recipe")
        if any(step.region == "$defect" for step in self.recipe) and not self.defects:
            raise ValueError("region '$defect' needs defect conditions")
        for field, condition in self.metrics.items():
            unknown_ops = sorted(set(condition) - set(METRIC_OPERATORS))
            if not condition or unknown_ops:
                raise ValueError(f"metrics.{field}: operators must be in {METRIC_OPERATORS}")
        return self


class Principle(BaseModel):
    """Một đoạn nguyên lý (tầng B), tách từ file markdown theo tiêu đề '##'."""

    id: str
    title: str
    text: str
    tags: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class KnowledgeBase(BaseModel):
    """Toàn bộ tri thức đã nạp và kiểm tra."""

    cards: List[PlaybookCard]
    principles: List[Principle]

    @model_validator(mode="after")
    def _unique_ids(self) -> "KnowledgeBase":
        """id card và id nguyên lý không được trùng."""
        for kind, ids in (
            ("card", [card.id for card in self.cards]),
            ("principle", [principle.id for principle in self.principles]),
        ):
            duplicates = sorted({item for item in ids if ids.count(item) > 1})
            if duplicates:
                raise ValueError(f"duplicate {kind} ids {duplicates}")
        return self
