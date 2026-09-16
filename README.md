# 오목 강화학습 챌린지 — 프로젝트

8×8, 5목 규칙의 Gomoku에서 사전학습된 baseline(순수 MCTS / AlphaZero 스타일 신경망+MCTS)을
상대로 겨루는 개인 프로젝트 구현.

> **참고:** 원본 챌린지 문서는 `junxiaosong/AlphaZero_Gomoku` 저장소의 게임 엔진과
> 사전학습 가중치를 그대로 사용하는 것을 전제로 한다. 이 저장소는 처음에 비어 있었기
> 때문에, 그 설계를 참고하여 엔진(`game.py`, `mcts_pure.py`, `mcts_alphaZero.py`,
> `policy_value_net_pytorch.py`)을 새로 작성했고, `best_policy_8_8_5.model`도
> 원본 다운로드 없이 `train.py`로 직접 self-play 학습시켜 만들었다. 따라서 baseline의
> 절대 강도는 원본 프로젝트와 다를 수 있으며, 실제 수치는 `evaluate.py` 출력 기준으로만
> 판단해야 한다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `game.py` | Board/Game 엔진 (8×8, 5목, 최대 64수) — 손대지 않음 |
| `mcts_pure.py` | 순수 MCTS (신경망 없음) — `pure_mcts` baseline에 사용 |
| `mcts_alphaZero.py` | PUCT 기반 MCTS (정책·가치망 결합) — `alphazero` baseline과 self-play 학습에 사용 |
| `policy_value_net_pytorch.py` | PyTorch 정책·가치망 (conv trunk + policy/value head) |
| `baseline_bot.py` | 두 baseline(`pure_mcts`, `alphazero`)을 노출 — 손대지 않음 |
| `train.py` | Self-play 학습 파이프라인. baseline 가중치 생성 및 참가자 자율 학습용 출발점 |
| `evaluate.py` | 로컬 평가 스크립트 (baseline 2종 × 각 색 교대) — 손대지 않음 |
| `league.py` | 참가자 간 라운드로빈 리그 스크립트 — 손대지 않음 |
| `student_agent.py` | **제출 대상.** 내 에이전트 구현 |
| `best_policy_8_8_5.model` | `alphazero` baseline이 로드하는 사전학습 가중치 (자체 self-play로 생성) |

## 내 에이전트 (`student_agent.py`) 접근 방법

강화학습 self-play 대신 **"규칙 기반 위협 감지 + 알파-베타 탐색"** 조합을 선택했다:

1. **정적 평가 함수**: 보드의 각 4방향 라인에서 연속 돌 개수(`length`)와 양끝이 열려
   있는지(`open_ends`, 0/1/2)를 계산해 점수화한다 (열린 4 ≈ 확정승, 열린 3은 강한 압박,
   막힌 패턴은 감점 없음 등 표준 오목 휴리스틱 테이블).
2. **후보 수 가지치기**: 이미 둔 돌 주변 반경 2칸 이내의 빈 칸만 후보로 삼아 분기 수를
   줄인다.
3. **즉시 승리/방어 체크**: 루트에서 내가 바로 이길 수 있으면 그 수, 아니면 상대가
   바로 이길 수 있는 수를 우선 차단한다.
4. **Iterative deepening negamax + alpha-beta**: 후보 수를 휴리스틱 점수로 정렬한 뒤
   깊이를 2→5까지 점진적으로 늘려가며 탐색하고, 매 수마다 시간 예산 내에서 최선 결과를
   채택한다.
5. **시간 관리**: 게임당 누적 300초 예산을 "남은 내 차례 수"로 나눠 매 수의 탐색
   시간을 동적으로 배분한다 (평가 스크립트의 sudden-death 몰수패 규정을 지키기 위함).

이 방식을 택한 이유:
- 학습이 전혀 필요 없어 "학습 wall-clock ≤ 4시간" 제약을 자동으로 만족한다.
- 완전히 결정적이고 디버깅이 쉬우며, self-play RL보다 훨씬 적은 개발/컴퓨트 비용으로
  `pure_mcts`(n_playout=1000)를 안정적으로 이긴다 (아래 결과 참고).
- 코드가 예측 가능해서 시간 예산 위반(몰수패) 위험이 낮다.

### Baseline 자체 학습 (`train.py`)

`alphazero` baseline이 로드할 `best_policy_8_8_5.model`은 `train.py`로 직접 self-play
학습시켜 만들었다 (원본 가중치를 내려받을 방법이 없었기 때문). 8×8 conv net
(3×conv + policy/value head), n_playout=400 self-play, 대칭 8배 데이터 증강, 주기적으로
`pure_mcts`와 대국해 승률이 오를 때만 최고 모델을 갱신하는 표준 AlphaZero self-play
루프를 그대로 구현했다.

## 실행 방법

```bash
pip install -r requirements.txt

# 파이프라인 빠른 확인
python evaluate.py --agent student_agent --games 5

# 전체 평가 (pure_mcts 50판 + alphazero 50판, 색 교대)
python evaluate.py --agent student_agent

# 리그 (여러 팀)
python league.py --teams student_agent other_team --games 20

# baseline 가중치 재학습 (선택)
python train.py --game-batch-num 1200 --time-budget-min 75
```

## 결과

`python evaluate.py --agent student_agent` 전체 실행 결과 (100판, seed 고정, 재현 가능):

```
pure_mcts  | games=50   W=50  L=0   D=0   win_rate=1.000  (black: 25/25 , white: 25/25)
alphazero  | games=50   W=50  L=0   D=0   win_rate=1.000  (black: 25/25 , white: 25/25)

OVERALL    | games=100  W=100 L=0   D=0   win_rate=1.000
```

- 몰수패(시간 초과) 0건, 무승부 0건 — 100판 전부 정상 종료.
- `alphazero` baseline은 `train.py`로 자체 self-play 학습한 모델(75분 시간 예산 내
  380 self-play batch, pure_mcts n_playout=200 상대 평가 승률 0.875에서 수렴)이라
  원본 챌린지가 상정하는 만큼 강하지 않을 가능성이 크다. 즉 100% 승률은 (a) 규칙 기반
  탐색 에이전트가 두 baseline에 비해 실제로 강하다는 것과 (b) 자체 학습한
  `alphazero` baseline이 완전한 원본 학습량에는 못 미친다는 것, 두 요인이 함께
  작용한 결과로 해석해야 한다. 원본 사전학습 가중치를 구해서 교체하면 `alphazero`
  baseline이 더 강해지고 승률이 낮아질 수 있다.

## 제약 준수

- 학습 wall-clock: 0시간 (학습 불필요, 규칙 기반 탐색) — baseline 자체 학습(`train.py`,
  약 1~1.5시간)은 참가자 제출물이 아니라 프로젝트 인프라 준비 과정이므로 4시간 제약과
  무관.
- 가중치 크기: `student_agent.py`는 별도 가중치 파일이 없음 (0 MB).
- 게임당 300초 누적 시간 관리: `student_agent.py` 내부에서 자체적으로 시간 예산을
  추적하며 탐색 깊이를 조절한다.
