# 🤖 FD-MADDPG: Fractional Delay-Aware Multi-Agent Reinforcement Learning

> **Novel extension of DAMA-DDPG to handle non-integer timestep delays via Virtual Effective Actions (VEA)**

---

## 🎯 TL;DR

We extended delay-aware MARL from **integer-only delays** → **fractional/continuous delays**, enabling discrete RL systems to approximate continuous-time behavior. Our method performs **on par with integer-delay baselines** while handling realistic, non-integer latencies.

---

## 🧠 The Problem

| Existing Approaches | Our Solution |
|---------------------|--------------|
| DAMA-DDPG handles only integer delays (1, 2, 3...) | Supports fractional delays (1.7, 2.3, etc.) |
| Real-world systems have continuous latencies | Linear interpolation bridges discrete timesteps |
| No mechanism to approximate between timesteps | Virtual Effective Actions (VEA) blend past actions |

---

## 💡 Key Innovation: Virtual Effective Actions

For a fractional delay `d = I + f` where `f ∈ [0,1)`:

```
ã_t = (1 - f) · a_{t-I} + f · a_{t-(I+1)}
```

**Translation:** Instead of picking ONE past action, we *blend* two consecutive actions proportionally — simulating what would happen if the action arrived "in between" timesteps.

---

## 📊 Results Summary

| Model | Delay | Performance |
|-------|-------|-------------|
| Delay-Unaware MADDPG | — | ❌ Unstable, slow convergence |
| Integer-Delay MARL | 1, 2 | ✅ Stable, good rewards |
| **FD-MADDPG (Ours)** | 1.7, 2.7 | ✅ **Matches integer-delay performance** |

*Tested on PettingZoo's `simple_spread_v3` with 3 cooperative agents over 10K-30K episodes.*

---

## 🔧 Implementation Highlights

- **Extended MADDPG** with action buffers for delay tracking
- **Custom environment wrappers** — migrated from deprecated MPE to PettingZoo's latest API
- **Linear interpolation module** for computing virtual effective actions
- **Centralized Training, Decentralized Execution (CTDE)** paradigm

---

## 📈 Training Curves

- **Non-integral delay (ours)** performs **on par with integral delay-aware** models
- Both delay-aware approaches significantly outperform delay-unaware baseline
- Delay-unaware training shows **unstable learning** with high variance and slower convergence

---
---

## 🔑 Key Takeaways

- ✅ Fractional delay handling via interpolation **works** — no performance degradation vs integer delays
- ✅ Provides smooth bridge from **discrete → continuous** delay modeling
- ✅ Drop-in compatible with existing MADDPG implementations
- ✅ Updated for **PettingZoo v3** (latest multi-agent env standard)

---

## 📚 References

- Chen et al. (2020) — *Delay-Aware Multi-Agent RL for Cooperative and Competitive Environments*
- Lowe et al. (2017) — *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*
- Hou & Phoha (2010) — *Control Delay in RL for Real-Time Dynamic Systems*

---


---

*Bridging the gap between theoretical discrete-time MARL and real-world continuous delays.*
