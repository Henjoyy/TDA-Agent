"""User Journey 입력 파서 - JSON/CSV 파일을 UserJourney 모델로 변환"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from tad_mapper.models.journey import TaskStep, UserJourney


class JourneyParser:
    """JSON 및 CSV 형식의 User Journey 파일 파서"""

    def parse(self, filepath: str | Path) -> UserJourney:
        """파일 확장자에 따라 자동으로 파서 선택"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            return self.parse_json(path)
        elif suffix == ".csv":
            return self.parse_csv(path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix} (지원: .json, .csv)")

    def parse_json(self, filepath: str | Path) -> UserJourney:
        """JSON 파일 파싱"""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return UserJourney.model_validate(data)

    def parse_csv(self, filepath: str | Path) -> UserJourney:
        """
        CSV 파일 파싱.

        CSV 형식 (헤더 필수):
        id, name, description, actor, input_data, output_data, dependencies, tags

        - input_data, output_data, dependencies, tags: 세미콜론(;)으로 구분
        """
        steps: list[TaskStep] = []

        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = TaskStep(
                    id=row["id"].strip(),
                    name=row["name"].strip(),
                    description=row.get("description", "").strip(),
                    actor=row.get("actor", "user").strip(),
                    input_data=self._split_list(row.get("input_data", "")),
                    output_data=self._split_list(row.get("output_data", "")),
                    dependencies=self._split_list(row.get("dependencies", "")),
                    tags=self._split_list(row.get("tags", "")),
                )
                steps.append(step)

        # CSV에서는 여정 메타데이터를 파일명에서 추출
        stem = Path(filepath).stem
        return UserJourney(
            id=stem,
            title=stem.replace("_", " ").title(),
            steps=steps,
        )

    @staticmethod
    def _split_list(value: str) -> list[str]:
        """세미콜론으로 구분된 문자열을 리스트로 변환"""
        if not value or not value.strip():
            return []
        return [item.strip() for item in value.split(";") if item.strip()]
