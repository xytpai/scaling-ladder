## 从零写一个PPO强化学习训练吃豆人

这篇文章将带你从最基础的 Policy Gradient 理解开始，逐步拆解 PPO 的收敛原理与设计动机，最后透过吃豆人这个直观Demo，看到强化学习从理论到实战的完整闭环。

### 1. Policy Gradient

对于 Agent 模型 $\theta$ ，我们要优化的对象 $R_{\theta}$ 是其所有走出路径的 Reward 之和，可以由以下公式描述：

$$
R_\theta = \sum_\tau reward(\tau)p_\theta(\tau)\\
$$

其中 $p_{\theta}(\tau)$ 表示该模型走出路径 $\tau$ 的概率。确定目标函数后，我们可以使用 Gradient Ascent 来优化 $\theta$ ，因此先初步展开 Gradient 的算式：

$$
\nabla R_\theta = \sum_\tau reward(\tau) \nabla p_\theta(\tau) \\ = \sum_\tau reward(\tau) p_\theta(\tau) \frac{\nabla p_\theta(\tau)} {p_\theta(\tau)} \\ = \sum_\tau reward(\tau) p_\theta(\tau) \nabla logp_\theta(\tau) \\ = E_{\tau -p_\theta(\tau)}[reward(\tau)\nabla logp_\theta(\tau)] \\ \approx \frac1n \sum_{\tau=1}^n reward(\tau)\nabla logp_\theta(\tau)\\
$$

以上推式很容易得到，重要的问题是 $\nabla logp_\theta(\tau)$ 是什么？或者说，如何将 Gradient 传导至模型参数 $\theta$ ? 首先我们逐步拆解 $p_{\theta}(\tau)$ :

$$
p_{\theta}(\tau)=p(s_{\tau1})p_\theta(a_{\tau1}|s_{\tau1})p(s_{\tau2}|s_{\tau1},a_{\tau1})p_\theta(a_{\tau2}|s_{\tau2})...\\
$$

上述式子中， $s$ 表示环境状态， $a$ 表示模型动作， $p(s_{\tau 1})$ 是这条路径的初始环境状态产生的概率， $p_{\theta}(a_{\tau 1}|s_{\tau 1})$ 表示在初始环境状态下模型产生动作 $a_{\tau 1}$ 的概率， $p(s_{\tau2}|s_{\tau1},a_{\tau1})$ 表示在产生 $s_{\tau 1}$ 环境状态以及模型产生 $a_{\tau 1}$ 动作后，产生新环境状态 $s_{\tau 2}$ 的概率（即模型第一步干预后）， $p_{\theta}(a_{\tau 2} | s_{\tau 2})$ 即新环境状态出现后模型产生动作 $a_{\tau 2}$ 的概率，以此类推。那么，当我们计算 Gradient 时，由于 Log 函数变成了加法，和 $\theta$ 的无关项都可忽略，梯度推导式又变成了如下：

$$
\nabla R=\frac1n \sum_{\tau=1}^n reward(\tau) \sum_{t=1}^{T_\tau}   \nabla logp_{\theta}(a_{\tau t}|s_{\tau t}) \\ =\frac1n \sum_{\tau=1}^n \sum_{t=1}^{T_\tau}  reward(\tau)  \nabla logp_{\theta}(a_{\tau t}|s_{\tau t}) \\
$$

以上就是 Policy Gradient 的最终算式。从该算式不难看出：如果某条路径总体是正反馈，那该条路径上每一步的环境状态对应的模型动作都会无脑给予正反馈，否则都会无脑惩罚，当采集的路径足够多后，模型就期望能收敛。而 $p_{\theta}(a|s)$ 其实就是一个神经网络的对应动作 channel 在输入为 s 情况下的输出值。

### 2. Proximal Policy Optimization (PPO)

如果直接使用 Policy Gradient 进行强化学习，有一个致命缺陷：模型必须不断与环境做互动来得到Reward并学习，这非常低效。打个比方，如果你需要从0开始搭建人类全部的科技树这太难了，但如果有一堆人已经帮你试错了所有的坑并为你总结出了各个领域的精华 (即采样环境收集反馈)，那你再统筹一下别人的经验就变得方便多了，PPO 就是做这个事情（学习他人）。

我们直接推导到最后：

$$
\nabla R_\theta = \sum_\tau reward(\tau) p_\theta(\tau) \nabla logp_\theta(\tau) \\ = \sum_\tau reward(\tau) p_{\theta'}(\tau) \frac{p_{\theta}(\tau)}{p_{\theta'}(\tau)} \nabla logp_\theta(\tau) \\  = E_{\tau -p_{\theta'}(\tau)}[\frac{p_{\theta}(\tau)}{p_{\theta'}(\tau)} reward(\tau)\nabla logp_\theta(\tau)] \\  \approx \frac1n \sum_{\tau=1}^n \frac{p_{\theta}(\tau)}{p_{\theta'}(\tau)} reward(\tau)\nabla logp_\theta(\tau),\ 用\theta'采样\\  =\frac1n \sum_{\tau=1}^n \frac{ p(s_{\tau1})p_\theta(a_{\tau1}|s_{\tau1})p(s_{\tau2}|s_{\tau1},a_{\tau1})p_\theta(a_{\tau2}|s_{\tau2})... }{ p(s_{\tau1})p_{\theta'}(a_{\tau1}|s_{\tau1})p(s_{\tau2}|s_{\tau1},a_{\tau1})p_{\theta'}(a_{\tau2}|s_{\tau2})... } reward(\tau) \sum_{t=1}^{T_\tau}   \nabla logp_{\theta}(a_{\tau t}|s_{\tau t}) \\  \approx \frac1n \sum_{\tau=1}^n \sum_{t=1}^{T_\tau} \frac{p_{\theta}(a_{\tau t}|s_{\tau t})}{p_{\theta'}(a_{\tau t}|s_{\tau t})} reward(\tau) \nabla logp_\theta(a_{\tau t}|s_{\tau t})  \\
$$

从推导过程可以看出，现在只需使用另一个模型 $\theta'$ 进行路径采样，就能直接计算本模型 $\theta$ 的梯度。这其中最有意思的是 $\frac {p_\theta(a|s)}{p_{\theta'}(a|s)}$ 这一项，如果代理模型相比本模型在某一状态的动作反馈度之比高很多，那就会极致地压低这一sample的训练影响，相反，如果本模型相对高则会增强。这里或多或少传达出一些哲学思辩：当我们发现某些人在某环境下相对高概率做某事，会觉得这是一种独特于他的特例从而降低学习他的比重；同时当我们在某环境下相对高概率做某事时，一旦发现其他人一不小心（即低概率）做了相同的事情并收到反馈，那我们也会对这一反馈记忆犹新。也即 Agent 学习它认为重要的事情 (i.e. Important Sampling)。

用以上推导的梯度公式进行学习(Off-Policy)会产生一个问题，如果代理模型的与本模型差异过大，可能采用到的路径相对本模型来说是基本不会走到的，这会极大降低学习效率，因此会有一个正则项来(KL散度)惩罚代理模型的差异，因此我们可以得到 PPO 的标准公式：

$$
\nabla J_{PPO}^{\theta'}(\theta) = \\\frac1n \sum_{\tau=1}^n \sum_{t=1}^{T_\tau} \frac{p_{\theta}(a_{\tau t}|s_{\tau t})}{p_{\theta'}(a_{\tau t}|s_{\tau t})} reward(\tau) \nabla logp_\theta(a_{\tau t}|s_{\tau t})  \\ - \beta \nabla KL(\theta, \theta') \\
$$

我们再对上面这个式子进行化简并裁，得到优化后的PPO公式：

$$
J_{PPO}^{\theta'} = \sum_{(s, a)}min(\\ \frac{p_\theta(a|s)}{p_{\theta'}(a|s)}A^{\theta'}(s, a),\\ clip(\frac{p_\theta(a|s)}{p_{\theta'}(a|s)}, 1-\epsilon, 1+\epsilon)A^{\theta'}(s, a)\\ ) - \beta KL(\theta, \theta') \\
$$

上式中 $A^{\theta'}(s, a)$ 表示当在环境状态 $s$ 下代理模型产生 $a$ 动作能在未来获取多少Reward总和(再减去Baseline)，也可以笼统称为优势函数。

笔者写了一个简单的强化学习PPO训练吃豆人Demo (仅需要CPU即可训练)，链接：https://github.com/xytpai/policy-gradient-demo。

以下代码片段可以清晰地体现PPO的训练过程：

```python
import torch
import env
import model
import matplotlib.pyplot as plt


def main(
        batch_size=4,
        height=12,
        width=12,
        num_eat=20,
        nstep=50,
        nepisode=10000,
        gamma=0.9,
        lr=1e-3,
        clip_epsilon=0.2):
    
    agent = model.PolicyModel(height, width)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    overall_reward_list = []
    for episode in range(nepisode):
        
        playground = env.PlayGround(batch_size, height, width, num_eat)
        playground.set_random()

        observe_actions = []
        observe_log_probs = []
        observe_states = []
        observe_rewards = []

        with torch.no_grad():
            for step in range(nstep):
                state = playground.get_space()
                action_probs = agent(state)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                rewards = playground.interact(model.decode_action(action))
                observe_actions.append(action.clone())
                observe_log_probs.append(log_prob.clone())
                observe_states.append(state.clone())
                observe_rewards.append(rewards.clone())
            
            discounted_rewards = []
            cumulative_reward = 0
            for r in reversed(observe_rewards):
                cumulative_reward = r + gamma * cumulative_reward
                discounted_rewards.insert(0, cumulative_reward)
            discounted_rewards = torch.stack(discounted_rewards, dim=-1) # b, step
            discounted_rewards = (discounted_rewards - discounted_rewards.mean(dim=1, keepdim=True)
                                ) / (discounted_rewards.std(dim=1, keepdim=True) + 1e-6)
            observe_log_probs = torch.stack(observe_log_probs, dim=-1) # b, step
            observe_rewards = torch.stack(observe_rewards, dim=-1) # b, step

        actual_log_probs = []
        entropy_loss = 0
        for state, action in zip(observe_states, observe_actions):
            action_probs = agent(state)
            dist = torch.distributions.Categorical(action_probs)
            entropy_loss += dist.entropy().mean()
            log_prob = dist.log_prob(action)
            actual_log_probs.append(log_prob)
        actual_log_probs = torch.stack(actual_log_probs, dim=-1) # b, step

        ratio = torch.exp(actual_log_probs - observe_log_probs.detach())
        loss = - torch.min(
            ratio * discounted_rewards, 
            torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon
        ) * discounted_rewards).sum(dim=1).mean() - 0.01 * entropy_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Show demo


if __name__ == '__main__':
    main()


```

### 3. Group Relative Policy Optimization (GRPO)

这是一种PPO的改进，对一个输入问题 $q$ ，采集代理模型的多种输出序列 $\{o1, o2, ..., o_G\}$ 并更新本模型，其优化的目标函数如下:

$$
J_{GRPO}^{\theta'} = \frac 1G \sum_{i=1}^Gmin(\\ \frac{p_\theta(o_i|q)}{p_{\theta'}(o_i|q)}A^{\theta'}(q,o_i),\\ clip(\frac{p_\theta(o_i|q)}{p_{\theta'}(o_i|q)}, 1-\epsilon, 1+\epsilon)A^{\theta'}(q,o_i)\\ ) - \beta KL(\theta, \theta') \\
$$

这其中优势函数是经过标准化的：

$$
A_i=\frac{r_i - mean(\{r_1, r2,...,r_G\})}{std(\{r1,r2,...,r_G\})}\\
$$

上式中 $r_i$ 表示某条输出的Reward是多少。

### 4. 语言模型的 Reward 怎么给

对于确定性问题，一定是有确定性答案的，但是对于非确定性问题（比如写作）很难给出标准的反馈。确定性问题能一定程度上反应出逻辑能力，因此，我们可以用确定性问题的 Question&Answer 来作为模型的训练资料，也就是说，我们可以只通过强化学习训练确定性问题（比如选择题/代码生成题)。确定性问题的反馈很容易获得，如选择题如果答对了就直接给分，答错扣分。
像 DeepSeek 这类自然语言交流模型会在问答时给出思维链 (Chain-Of-Thought)，这样能自我推断出更准确的回答，像是人类思考的过程。为了让机器获得这种能力，可以引入思维链模板与一个匹配度的Reward。比如我们的问题是“你是谁”，那我们输入到模型中的数据就是：

```txt
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: 你是谁? Assistant:
```

然后模型从 Assistant: 后开始接字。我们可以直接使用 ChatGPT 做实验，它的输出结果是:

```txt
<think>用户正在询问我的身份。我需要简单介绍自己并表明我是一个AI助手。</think>
<answer>我是你的AI助手，可以帮助你回答问题、解决问题或提供建议。</answer>
```

以上就是一个简单的思维链输出。
