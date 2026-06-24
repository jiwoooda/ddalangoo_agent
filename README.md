# 딸랑구 — Voice-First AI Shopping Assistant for Elderly

> 고령 사용자를 위한 음성 기반 멀티에이전트 쇼핑 어시스턴트  
> Graduation Project · Sookmyung Women's University · 2026

## Overview

딸랑구 is a **voice-first shopping assistant** designed for elderly users who face difficulty navigating modern e-commerce interfaces. Instead of complex UI, users speak naturally — the system handles intent understanding, product search, and payment through a multi-agent pipeline.

**Core design principles:**
- Elderly-friendly response generation (short sentences, honorifics, confirmation-heavy flow)
- 3-stage human-in-the-loop (HITL) verification before payment
- Proactive re-order suggestion based on purchase history

## Architecture

```
User Input (Voice / Text)
    └─ IntentAgent (GPT-4o-mini)        Intent classification
         └─ Router (LangGraph conditional edge)
              ├─ ContextAgent            Purchase history & preference lookup
              │    └─ ProductAgent       Product search & ranking
              │         └─ ResponseAgent Elderly-friendly response generation
              ├─ ReorderAgent            Re-purchase history search
              └─ PaymentSubgraph         5-step payment flow
```

**Tech stack:** LangGraph · Claude (Anthropic) · GPT-4o-mini (OpenAI) · MCP Tools · Python 3.11

## Demo

```
나:     딸기 사줘
딸랑구: 설향 딸기 500g, 12,900원이에요. 주문할까요?
나:     2개요
딸랑구: 딸기 2개 담았어요! 결제할까요?
나:     결제할게요
딸랑구: 총 25,800원이에요. 네이버로 결제할까요?
나:     응
딸랑구: 서울 강남구...로 보낼게요. 맞으시죠?
나:     맞아요
딸랑구: 비밀번호 입력해주세요!
나:     1234
딸랑구: 완료! 주문이 접수됐어요.
```

```
# 재구매 흐름
나:     저번에 산 계란 다시 줘
딸랑구: 풀무원 유정란 10구, 4,900원이에요. 다시 주문할까요?
```

## Getting Started

**Requirements:** Python 3.11+, Anthropic API key, OpenAI API key

```bash
git clone https://github.com/jiwoooda/ddalangoo_agent
cd ddalangoo-agent
python -m venv .venv
.venv/bin/pip install -e .          # macOS/Linux
# .\.venv\Scripts\pip install -e .  # Windows
```

```bash
cp .env.example .env
# .env에 API 키 입력
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
```

```bash
python main.py                  # 기본 실행
python main.py --user user_001  # 구매이력 있는 유저 (재구매 기능)
python main.py --demo           # API 키 없이 구조 검증
python tests/test_e2e.py        # E2E 테스트
```

## Test Users

| User ID | Name | Purchase History |
|---------|------|-----------------|
| user_test | 테스트유저 | — |
| user_001 | 김영희 (60대) | 딸기, 계란 |
| demo | 데모유저 (40대) | 참기름 |

## Author

**Jiwoo Won** · [jiow2003@sookmyung.ac.kr](mailto:jiow2003@sookmyung.ac.kr) · [jiwoooda.github.io](https://jiwoooda.github.io)
