# 딸랑구 쇼핑 어시스턴트 — CrewAI 버전

고령 사용자를 위한 음성 기반 쇼핑 어시스턴트 (CrewAI 구현)

---

## 실행 전 필요한 것

- Python 3.11 이상
- Anthropic API 키 (Claude 사용)
- OpenAI API 키 (GPT-4o-mini 의도 분류 사용)

---

## 실행 방법 (순서대로)

### 1단계 — 가상환경 생성 및 패키지 설치

```bash
cd ddalangoo-crewai

python -m venv .venv
```

**Windows:**
```bash
.\.venv\Scripts\pip install -e .
```

**macOS / Linux:**
```bash
./.venv/bin/pip install -e .
```

---

### 2단계 — API 키 설정

`.env.example` 파일을 복사해서 `.env` 파일을 만들고, API 키를 입력합니다.

**Windows:**
```bash
copy .env.example .env
```

**macOS / Linux:**
```bash
cp .env.example .env
```

`.env` 파일을 열어 아래와 같이 수정합니다:

```
ANTHROPIC_API_KEY=sk-ant-여기에본인키입력
OPENAI_API_KEY=sk-proj-여기에본인키입력
```

---

### 3단계 — 실행

**Windows:**
```bash
# 대화형 실행
.\.venv\Scripts\python.exe main.py

# 구매이력 있는 유저로 실행 (재구매 기능 포함)
.\.venv\Scripts\python.exe main.py --user user_001

# E2E 자동 테스트
.\.venv\Scripts\python.exe e2e_test.py
```

**macOS / Linux:**
```bash
./.venv/bin/python main.py
./.venv/bin/python main.py --user user_001
./.venv/bin/python e2e_test.py
```

> **가상환경을 활성화한 경우** `.\.venv\Scripts\activate` (Windows) 또는 `source .venv/bin/activate` (macOS/Linux) 실행 후에는 `python main.py` 만 입력해도 됩니다.

---

## 테스트 유저

| 유저 ID | 구매이력 | 사용 방법 |
|---------|---------|----------|
| `user_test` (기본) | 없음 | `python main.py` |
| `user_001` | 딸기, 계란 (재구매 가능) | `python main.py --user user_001` |

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
나:    응 그거
딸랑구: 계란 몇 개 드릴까요?
나:    1개요
딸랑구: 계란 1개 담았어요! 결제할까요?
...
```

**취소**
```
나:    우유 사줘
딸랑구: 서울우유 1L, 2,800원이에요. 주문할까요?
나:    그만할게요
딸랑구: 쇼핑을 종료했어요. 또 필요하시면 말씀해 주세요.
```

종료: 대화 중 `exit` 입력 또는 `Ctrl+C`

---

## 검색 가능한 상품 키워드

`딸기` / `계란` / `우유` / `사과` / `참기름` / `운동화`

그 외 키워드는 기본 상품 1종이 반환됩니다.

---

## Crew 구성

```
사용자 입력
    └─ IntentCrew (GPT-4o-mini)        의도 분류
         └─ Router (Python 함수)
              ├─ BuyCrew               구매 흐름
              │    ├─ ContextLoader     구매이력 · 선호도 조회
              │    ├─ ProductSearcher   상품 검색 · 랭킹
              │    └─ ResponseComposer  어르신 친화 응답 생성
              ├─ ReorderCrew           재구매 흐름
              │    ├─ ReorderResolver   이력 탐색
              │    └─ ResponseComposer  응답 생성
              └─ PaymentCrew           결제 5단계 처리
                   └─ PaymentProcessor 장바구니 · 주문 · 배송지 확인
```
