"""
TAD-Mapper: 자동 에이전트 설계 및 매핑 시스템
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent

# ── Gemini API 설정 ──────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
EMBEDDING_MODEL_CANDIDATES: list[str] = [
    m.strip()
    for m in os.getenv(
        "EMBEDDING_MODEL_CANDIDATES",
        "gemini-embedding-001,models/text-embedding-004,text-embedding-004",
    ).split(",")
    if m.strip()
]

# ── Unit Agent 정의 파일 ─────────────────────────────────────
UNIT_AGENTS_CONFIG: Path = PROJECT_ROOT / "config" / "unit_agents.yaml"

# ── TDA 파라미터 ─────────────────────────────────────────────
TDA_N_INTERVALS: int = int(os.getenv("TDA_N_INTERVALS", "10"))
TDA_OVERLAP_FRAC: float = float(os.getenv("TDA_OVERLAP_FRAC", "0.3"))
TDA_MIN_SAMPLES: int = int(os.getenv("TDA_MIN_SAMPLES", "2"))

# ── 출력 경로 ────────────────────────────────────────────────
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── API 서버 설정 ────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
ROUTER_MAX_FALLBACK_RATIO: float = float(os.getenv("ROUTER_MAX_FALLBACK_RATIO", "0.2"))
ROUTER_MIN_EMBED_CALLS: int = int(os.getenv("ROUTER_MIN_EMBED_CALLS", "5"))
ROUTE_MIN_CONFIDENCE: float = float(os.getenv("ROUTE_MIN_CONFIDENCE", "0.35"))


def validate_config() -> None:
    """필수 설정값 검증"""
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY가 설정되지 않았습니다. "
            ".env 파일에 GEMINI_API_KEY=your_key 를 추가해주세요."
        )
