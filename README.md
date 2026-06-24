# 딸랑구 쇼핑 어시스턴트 — LangGraph 버전

고령 사용자를 위한 음성 기반 쇼핑 어시스턴트 (LangGraph 구현)

---

## 환경 요구사항

- Python **3.11 이상**
- API 키: Anthropic, OpenAI

---

## 의존성 설치

```bash
cd ddalangoo-agent
python -m venv .venv

# Windows
.\.venv\Scripts\pip install -e .

# macOS / Linux
./.venv/bin/pip install -e .
```

---

## 환경변수 설정

`.env.example`을 복사해 `.env` 파일을 생성하고 API 키를 입력합니다.

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

```
# .env
ANTHROPIC_API_KEY=sk-ant-여기에입력    # Claude (Product Agent) 사용
OPENAI_API_KEY=sk-proj-여기에입력      # GPT-4o-mini (Intent Agent) 사용
```

---

## 실행 명령어

```bash
# 대화형 실행 (기본 유저)
.\.venv\Scripts\python.exe main.py

# 구매이력 있는 유저 실행 (재구매 기능 포함)
.\.venv\Scripts\python.exe main.py --user user_001

# E2E 자동 테스트
.\.venv\Scripts\python.exe tests/test_e2e.py

# API 키 없이 구조 검증만
.\.venv\Scripts\python.exe main.py --demo
```

---

## 에이전트 구성

```
사용자 입력
    └─ IntentAgent (GPT-4o-mini)       의도 분류
         └─ Router (LangGraph 조건부 엣지)
              ├─ ContextAgent           구매이력 · 선호도 조회
              │    └─ ProductAgent      상품 검색 · 랭킹
              │         └─ ResponseAgent 어르신 친화 응답 생성
              ├─ ReorderAgent           재구매 이력 탐색
              └─ PaymentSubgraph        결제 5단계 처리
```

---

## 테스트 유저

| 유저 ID | 이름 | 구매이력 | 재구매 가능 상품 |
|---------|------|---------|----------------|
| `user_test` (기본) | 테스트유저 | 없음 | — |
| `user_001` | 김영희 (60대) | 딸기, 계란 | 딸기, 계란 |
| `demo` | 데모유저 (40대) | 참기름 | 참기름 |

---

## 검색 가능한 상품 키워드

`딸기` / `계란` / `우유` / `사과` / `참기름` / `운동화`  
그 외 키워드는 기본 상품 1종이 반환됩니다.

---

## 대화 예시

**구매 흐름**
```
나:    딸기 사줘
딸랑구: 설향 딸기 500g, 12,900원이에요. 주문할까요?
나:    2개요
딸랑구: 딸기 2개 담았어요! 결제할까요?
나:    결제할게요
딸랑구: 총 25,800원이에요. 네이버로 결제할까요?
나:    응
딸랑구: 서울 강남구...로 보낼게요. 맞으시죠?
나:    맞아요
딸랑구: 비밀번호 입력해주세요!
나:    1234
딸랑구: 완료! 주문이 접수됐어요.
```

**재구매 흐름** (`--user user_001` 실행 시)
```
나:    저번에 산 계란 다시 줘
딸랑구: 풀무원 유정란 10구, 4,900원이에요. 다시 주문할까요?
나:    응
딸랑구: 계란 몇 개 드릴까요?
나:    1개요
딸랑구: (결제 5단계 진행...)
```

**취소**
```
나:    우유 사줘
딸랑구: 서울우유 1L, 2,800원이에요. 주문할까요?
나:    그만할게요
딸랑구: 쇼핑을 종료했어요. 또 필요하시면 말씀해 주세요.
```

종료: 대화 중 `exit` 입력 또는 `Ctrl+C`
