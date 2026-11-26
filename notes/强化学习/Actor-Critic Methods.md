# Actor-Critic Methods(AC)



## What are  “actor” and “critic”?

- “actor” refers to policy update.
- “critic” refers to policy evaluation or value estimation.





---

## 1.QAC

- ==aiming to maximize $J(\theta)$==

At time step t in each episode , do:

​		generate $a_t$ following $\pi(a | s_t , \theta_t)$,observe $r_{t+1} , s_{t+1}$,and then generate $a_{t+1}$ following $\pi(a | s_{t+1} , \theta_t)$. 

​		critic(value update):

​		$w_{t+1} = w_t + \alpha_w[r_{t+1} + \gamma q(s_{t+1} , a_{t+1} , w_t) - q(s_t , a_t , w_t)] \nabla_w q(s_t , a_t , w_t)$

​		actor (policy update):

​		$\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta ln\pi(a_t|s_t , \theta_t)q(s_t , a_t , w_{t+1})$



---

## 2. A2C(advantage actor-critic)

==property: thee policy gradient is invariant to an additional baseline==

### $\nabla_\theta J(\theta) = E_{S \sim\eta , A \sim \pi}[\nabla_\theta ln\pi(A|S , \theta_t)q_\pi(S , A)] = E_{s \sim \eta , A \sim \pi}[\nabla_\theta ln\pi(A|S , \theta_t)(q_\pi (S,A) - b(S))]$



### First ,why is it valid?

$E_{S\sim\eta,A\sim\pi}[\nabla_\theta ln\pi(A|S,\theta_t)b(S)]$ 

$= \sum_{s\in S}\eta(s) \sum_{a\in\Alpha }\pi(a|s,\theta_t) \nabla_\theta ln\pi(a|s,\theta_t)b(s)$

$=\sum_{s\in S}\eta(s)b(s)\nabla_\theta \sum_{a\in A}\pi(a|s,\theta_t)$

$=\sum_{s\in S}\eta(s)b(s)\nabla_\theta 1$

$=0$



### Second , why is the baseline useful?

#### - E[X] is invariant to b(S),but var(X) is not invariant to b(S).

$X(S,A) = \nabla_\theta ln\pi(A|S,\theta_t)[q(S,A)-b(S)]$

$Var(X) = E[X^2] - (E(X))^2$

$E(X^2) = E[\vert\vert\nabla_\theta ln\pi\vert\vert ^2 (q(S,A)-b(S))^2]$

` q(S,A)-b(S) != 0`



At time step t in each episode , do:

​		generate $a_t$ following $\pi(a|s_t,\theta_t)$ and then observe $r_{t+1},s_{t+1}$.

​		TD error(advantage function):

​		$\delta_t = r_{t+1} +\gamma v(s_{t+1},w_t) - v(s_t,w_t)$

​		Critic(value update):

​		$w_{t+1} = w_t + \alpha_w\delta_t\nabla_w v(s_t,w_t)$

​		Actor(policy update):

​		$\theta_{t+1} = \theta_t + \alpha_\theta\delta_t\nabla_\theta ln\pi(a_t|s_t,\theta_t)$



---

## 3.Off-policy actor-critic

At time step t in each episode , do:

​		generate $a_t$ following $\beta(s_t)$ and then observe $r_{t+1},s_{t+1}$.

​		TD error(advantage function):

​		$\delta_t = r_{t+1} +\gamma v(s_{t+1},w_t) - v(s_t,w_t)$

​		Critic(value update):

​		$w_{t+1} = w_t + \alpha_w \frac{\pi(a_t|s_t,\theta_t)}{\beta(a_t|s_t)}\delta_t\nabla_w v(s_t,w_t)$

​		Actor(policy update):

​		$\theta_{t+1} = \theta_t + \alpha_\theta \frac{\pi(a_t|s_t,\theta_t)}{\beta(a_t|s_t)}\delta_t\nabla_\theta ln\pi(a_t|s_t,\theta_t)$



---

## 4.DPG(Deterministic actor-critic)

At time step t in each episode , do:

​		generate $a_t$ following $\beta$ and then observe $r_{t+1},s_{t+1}$.

​		TD error:

​		$\delta_t = r_{t+1} +\gamma q(s_{t+1},w_t) - q(s_t,a_t,w_t)$

​		Critic(value update):

​		$w_{t+1} = w_t + \alpha_w \delta_t\nabla_w v(s_t,a_t,w_t)$

​		Actor(policy update):

​		$\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \mu(s_t,\theta_t)(\nabla_a q(s_t,\theta_t)$

