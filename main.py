"""
딸랑구 쇼핑 어시스턴트 — Standalone CLI Entry Point.

사용법:
    python main.py
    python main.py --user user_001
    python main.py --user user_001 --thread my-thread --trace
    python main.py --demo               (LLM 없이 라우터/플로우 검증)
    python main.py --list-users         (사용 가능한 mock 사용자 목록)

환경변수 (.env 또는 shell):
    ANTHROPIC_API_KEY   Product Agent (Claude 사용)
    OPENAI_API_KEY      Intent Agent (GPT-4o-mini 사용)
    NAVER_CLIENT_ID     (선택) 실제 네이버 쇼핑 검색
    NAVER_CLIENT_SECRET (선택) 실제 네이버 쇼핑 검색
    META_MCP_SERVER_URL (선택) 원격 MCP 서버
    LOG_AGENT_TRACE     true/false (에이전트 추적 로그)
"""
import argparse
import uuid
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# ── API Key 경고 ─────────────────────────────────────────────────────
_missing = []
if not os.getenv("ANTHROPIC_API_KEY"):
    _missing.append("ANTHROPIC_API_KEY (Product Agent - Claude)")
if not os.getenv("OPENAI_API_KEY"):
    _missing.append("OPENAI_API_KEY (Intent Agent - GPT-4o-mini)")

if _missing:
    print("[경고] 다음 환경 변수가 설정되지 않았습니다:")
    for key in _missing:
        print(f"       - {key}")
    print("       .env 파일을 생성하거나 환경 변수를 설정하세요.")
    print("       LLM 호출이 필요한 에이전트가 동작하지 않을 수 있습니다.\n")


from src.graph.builder import build_graph
from src.state.schema import get_default_shopping_state
from src.utils.agent_logger import agent_logger


def run_session(user_id: str = "user_test", thread_id: str | None = None, trace: bool = False) -> None:
    """단일 대화 세션 실행 (CLI REPL)."""
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    agent_logger.start_session(session_id=thread_id, console=trace)

    print(f"\n{'='*52}")
    print(f"  딸랑구 쇼핑 어시스턴트 (Standalone)")
    print(f"  user: {user_id} | thread: {thread_id[:8]}...")
    if trace:
        print(f"  TRACE ON  →  {agent_logger.log_path}")
    print(f"  종료: 'exit' 또는 Ctrl+C")
    print(f"{'='*52}\n")

    # 검색 모드 표시
    if os.getenv("META_MCP_SERVER_URL"):
        print(f"  [검색] 원격 MCP 서버: {os.getenv('META_MCP_SERVER_URL')}")
    elif os.getenv("NAVER_CLIENT_ID"):
        print("  [검색] 네이버 쇼핑 API")
    else:
        print("  [검색] 내장 Mock 데이터 (NAVER_CLIENT_ID 미설정)")
    print()

    graph = build_graph()

    initial_state = get_default_shopping_state(user_id, session_id)
    initial_state["conversation_id"] = abs(hash(thread_id)) % 1_000_000
    graph.invoke(initial_state, config)

    while True:
        try:
            user_input = input("사용자: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[중단됨]")
            break

        if user_input.lower() in ("exit", "quit", "종료", "q"):
            print("[대화 종료]")
            break
        if not user_input:
            continue

        agent_logger.new_turn(user_input)
        graph.update_state(config, {"messages": [{"role": "user", "content": user_input}]})

        try:
            graph.invoke(None, config)
        except Exception as e:
            print(f"[그래프 오류] {e}")
            import traceback
            traceback.print_exc()
            break

        current = graph.get_state(config)
        messages = current.values.get("messages", [])
        response = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                response = msg["content"]
                break
            if getattr(msg, "type", None) == "ai":
                response = msg.content
                break

        stage = current.values.get("stage", "unknown")
        pending = current.values.get("pending_action")
        pending_str = f" | pending: {pending.get('type')}" if pending else ""

        print(f"\n딸랑구: {response}")
        print(f"  └─ stage={stage}{pending_str}\n")

        if not current.next:
            print("[세션 종료]")
            break
        if stage in ("completed", "failed"):
            break


def run_demo() -> None:
    """API key 없이 라우터/플로우만 검증하는 데모."""
    print("\n=== Demo Mode (LLM 없이 라우터/플로우/결제 검증) ===\n")

    from src.graph.router import route, after_respond

    # 1. Router 테스트
    print("─── Router 테스트 ───")
    cases = [
        {"intent": "buy",     "stage": "idle",              "confidence": 0.95, "needs_clarification": False},
        {"intent": "reorder", "stage": "idle",              "confidence": 0.90, "needs_clarification": False},
        {"intent": "confirm", "stage": "product_confirming","confidence": 0.95, "needs_clarification": False,
         "quantity": 2, "pending_action": {"type": "product_confirm"}},
        {"intent": "cancel",  "stage": "payment_processing","confidence": 0.95, "needs_clarification": False},
        {"intent": "unclear", "stage": "idle",              "confidence": 0.30, "needs_clarification": True},
        {"intent": "next",    "stage": "product_confirming","confidence": 0.95, "needs_clarification": False},
    ]
    for case in cases:
        state = get_default_shopping_state("user_test", "sess")
        state.update(case)
        result = route(state)
        intent = case['intent']
        stage = case['stage']
        print(f"  {intent:<20} + {stage:<22} → {result}")

    # 2. Context Agent 테스트 (mock DB, LLM 없이)
    print("\n─── Context Agent 테스트 (mock DB) ───")
    from src.agents.context_agent import get_recommendation_context
    ctx = get_recommendation_context("user_001", ["딸기"], intent="buy")
    print(f"  user_001 구매이력: {ctx.get('purchase_count')}건")
    print(f"  retrieval_mode: {ctx.get('retrieval_mode')}")
    print(f"  keyword_results: {len(ctx.get('keyword_results', []))}건")

    # 3. Mock Search 테스트
    print("\n─── Mock Search 테스트 ───")
    from src.tools.mock_search import search_products
    results = search_products("딸기", ["kurly", "naver"], condition="price_asc")
    print(f"  '딸기' 검색 결과: {len(results)}건")
    for r in results[:3]:
        print(f"    - {r['product_name']} {r['price']:,}원 ({r['platform']})")

    # 4. Reorder Agent 테스트
    print("\n─── Reorder Agent 테스트 ───")
    from src.agents.reorder_agent import _resolve_reorder_candidates
    reorder_result = _resolve_reorder_candidates("user_001", ["딸기"], "딸기")
    print(f"  resolution_type: {reorder_result.get('resolution_type')}")
    candidate = reorder_result.get("selected")
    if candidate:
        print(f"  selected: {candidate.get('product_name')} ({candidate.get('price_at_purchase', 0):,}원)")

    print("\n[Demo 완료] ✓")


def list_users() -> None:
    from src.tools.mock_tools import MOCK_USERS, MOCK_PURCHASE_HISTORY
    print("\n사용 가능한 Mock 사용자:")
    print(f"  {'user_id':<15} {'name':<12} {'age_group':<10} {'구매이력'}")
    print(f"  {'-'*50}")
    for uid, info in MOCK_USERS.items():
        history_count = len(MOCK_PURCHASE_HISTORY.get(uid, []))
        print(f"  {uid:<15} {info.get('name', ''):<12} {info.get('age_group', ''):<10} {history_count}건")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="딸랑구 쇼핑 어시스턴트 — Standalone LangGraph Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py                          # 기본 실행 (user_test)
  python main.py --user user_001          # 구매이력 있는 사용자
  python main.py --user demo --trace      # 에이전트 추적 로그 출력
  python main.py --demo                   # LLM 없이 플로우 검증
  python main.py --list-users             # mock 사용자 목록

Mock 상품 키워드: 딸기, 운동화, 참기름, 계란, 우유, 사과 (그 외 키워드도 기본 상품 반환)
""",
    )
    parser.add_argument("--user", default="user_test", help="사용자 ID (기본: user_test)")
    parser.add_argument("--thread", default=None, help="스레드 ID (기본: 자동 생성)")
    parser.add_argument("--trace", action="store_true", help="에이전트 추적 로그 터미널 출력")
    parser.add_argument("--demo", action="store_true", help="LLM 없이 플로우/라우터 검증 실행")
    parser.add_argument("--list-users", action="store_true", help="사용 가능한 mock 사용자 목록 출력")
    args = parser.parse_args()

    if args.list_users:
        list_users()
        return

    if args.demo or (not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY")):
        run_demo()
    else:
        run_session(user_id=args.user, thread_id=args.thread, trace=args.trace)


if __name__ == "__main__":
    main()
