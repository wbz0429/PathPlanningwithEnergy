# AI4S 优势/域相关性 deep-research(2026-07,103 agents)

## 裁决
对研究问题的诚实回答:所有旗舰AI4S系统(FunSearch、AlphaEvolve、Eureka、AI Scientist、POISE)的正结果都严格集中在"有精确可自动评测器 + 真实改进headroom"的域——数学/组合优化/代码/RL奖励——而且它们的方法在结构上就要求这样的域;这在多篇一手论文中被作者自己明确承认为"局限"(AlphaEvolve把需要人工实验的域"排除在scope之外";FunSearch把缺乏rich scoring的theorem-proving明确排除),没有任何先例把AI4S成功用于"无干净ground truth的噪声真实传感器数据"域产出新发现。因此你的框架在能耗建模+能量感知规划这个噪声封顶域产出"正确的null",是域的性质决定的、而非框架失败——这个论点有一手文献直接支撑(AlphaEvolve在50+数学问题上75%只是"match"即null型结果、20%才有真发现;FunSearch最难任务n=8只有4/140≈2.9%成功率),说明即使在理想可验证域,真发现也只发生在有headroom的少数问题上,"匹配/持平"本就是常态。关于贡献价值:独立评估已实证AI4S的overclaim/不可复现失败模式(Sakana AI Scientist:42%实验因编码错误失败、57%手稿含错误或幻觉数值、把micro-batching等prior art误判为novel),而ML社区已有专门发表null/负结果与批判性工作的一手可发表先例(ICML 2024 "Embracing Negative Results"位置论文、NeurIPS ICBINB工作坊、EMNLP Insights工作坊/期刊、以及2025年提议在ML顶会设立"Refutations and Critiques"track的位置论文)。诚实裁决:仅凭"证明该域AI4S只能产出正确null + 我们的安全机制阻止了朴素AI4S的overclaim"作为主贡献,方法学上站得住(负结果的价值取决于实验严谨性,这有一手支撑),但审稿人会质疑其足够性;强烈建议按你自己的直觉,同时展示框架在有headroom的域(规划器loop胜随机27%)能产出正值,以证明null是域的问题而非框架的问题——这个"域相关价值 = 需同时具备可验证评测器+真实headroom,噪声封顶域产出null是正确而非失败"的核心论点有直接文献支撑。

## 发现(逐条·三票验证)
### [1] high · 3-0 (合并claims 0,1,3,6)
所有旗舰AI4S系统在结构上都要求一个精确、自动、可机器评分的评测器,并且这被作者明确点名为方法的先决条件/局限——AI4S不能用于缺乏此评测器的域。
**证据**:FunSearch: 'pairing a pre-trained LLM ... with an automated evaluator, which guards against hallucinations'; 'We focus in this paper on problems admitting an efficient evaluate function.' AlphaEvolve自己声明: 'the user must provide a mechanism for automatically assessing generated solutions ... a function h mapping a solution to a set of scalar evaluation metrics',并把它列为'main limitation ... it puts tasks that require manual experimentation out of our scope.' Eureka依赖Isaac Gym仿真作为可验证fitness信号。这是研究问题所依赖的确切前提:AI4S结构性地需要可验证评测器,而噪声真实传感器域恰恰缺此。
**源**:https://www.nature.com/articles/s41586-023-06924-6 ; https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf ; https://arxiv.org/abs/2310.12931 ; https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/Mathematical-discoveries-from-program-search-with-large-language-models.pdf

### [2] high · 3-0 (合并claims 2,4,7,14,20,21)
所有旗舰AI4S的正结果都集中在'有精确可验证评测器 + 真实改进headroom'的类型域(数学/组合优化/代码/RL奖励),作者本人明确说成功被限于这些域;没有任何一篇把AI4S成功用于'无干净ground truth的噪声真实传感器数据'域产出新发现的先例。
**证据**:FunSearch真发现只在cap-set(20年来对渐近下界的最大改进,声称首个用LLM的科学发现)和online bin-packing(击败first-fit/best-fit)——都是精确可验证组合/数学域,论文中无任何噪声传感器数据发现的例子。AlphaEvolve: 'Because problems in mathematics, computer science, and system optimization typically permit automated evaluation metrics, our efforts on AlphaEvolve focus on these domains',并承认自然科学等只能部分仿真的域超出scope。Eureka: RL奖励设计,83%任务胜过人类专家、平均52%提升,全在NVIDIA Isaac Gym仿真基准。POISE (arXiv 2603.23951): 只在AIME/MATH等clean数学竞赛基准上产出真提升(weighted Overall 47.8→52.5)。跨5个系统一致:正结果=clean可验证评测器域,无噪声真实传感器成功先例。
**源**:https://www.nature.com/articles/s41586-023-06924-6 ; https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf ; https://arxiv.org/abs/2310.12931 ; https://arxiv.org/html/2603.23951 ; https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/Mathematical-discoveries-from-program-search-with-large-language-models.pdf

### [3] high · 3-0 (合并claims 5,15,19)
即使在理想的可验证评测器域,真正的新发现也只发生在有headroom的少数问题上——'匹配/持平'(null型结果)是常态,且成功频率高度域/任务相关。这直接支撑:在噪声封顶域产出null是域的性质,不是框架失败。
**证据**:AlphaEvolve在50+开放数学构造问题上: '在75%的情况下重新发现了已知最优构造(很多可能本就是最优),只在20%的情况下发现了可证明更好的新构造'——即真发现只是有headroom的少数,'匹配'是null型常态。FunSearch自己的可复现性统计暴露高隐藏失败率: 最难任务cap-set n=8只有4/140≈2.9%的run成功,而'可靠'的I(12,7)也只有60%的run找到full-size集——成功频率域/任务相关,旗舰结果是众多失败run的幸存者。
**源**:https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf ; https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/Mathematical-discoveries-from-program-search-with-large-language-models.pdf

### [4] high · 3-0 (合并claims 8,9,10,16,17)
独立(第三方)评估已实证朴素AI4S系统的真实失败率、不可复现性和overclaim失败模式——这正是可信框架的安全机制要阻止的失败模式。
**证据**:对Sakana AI Scientist的独立评估(Beel/Kan/Baumgart,已发表于ACM SIGIR Forum, DOI 10.1145/3769733.3769747): (1) 42%实验(5/12)因编码错误直接失败;(2) 57%手稿(4/7)含错误或幻觉数值(超参和性能指标有出入);(3) 把已确立的prior art误判为'novel'(如micro-batching for SGD),因为文献综述用keyword匹配且无匹配时默认novel=True。这是survivorship/cherry-picking/overclaim的直接文献证据,也说明旗舰论文倾向只报成功。
**源**:https://arxiv.org/abs/2502.14297 ; https://arxiv.org/html/2502.14297v2

### [5] high · 3-0 (合并claims 11,12,22,23,24)
'可信但产出null result'/负结果/批判性工作是ML与元科学圈内被接受的可发表贡献类型,存在专门venue和顶会一手先例——支持'贡献是校准/诚实报告负结果/防overclaim'而非'新发现'的定位。
**证据**:'Position: Embracing Negative Results in Machine Learning'(Karl et al., ICML 2024, PMLR 235:23256-23265)明确论证'predictive performance alone is not a good indicator for the worth of a publication',呼吁发表负结果以改善ML科研健康。存在专门venue: NeurIPS 'I Can't Believe It's Not Better!'(ICBINB)工作坊、EMNLP 'Insights from Negative Results in NLP'工作坊/期刊。另有14位ML研究者(Schaeffer, Koyejo, Donoho, Dodge等)的位置论文(arXiv 2506.19882)提议ML顶会设立专门的'Refutations and Critiques'(R&C)track,把批判/反驳prior research作为一等贡献,理由是错误/有缺陷/甚至造假研究会因peer review的可错性被顶会接受甚至highlight,而当前无系统纠错机制。
**源**:https://arxiv.org/abs/2406.03980 ; https://arxiv.org/html/2406.03980v1 ; https://arxiv.org/abs/2506.19882

### [6] high · 3-0 (claim 13)
负结果的价值取决于实验的方法学严谨性——价值来自正确执行而非失败本身。这直接支撑'一个受控的、安全机制保护的null是贡献,而草率的失败不是'。
**证据**:ICML 2024位置论文: 'If experimental design was sound, analysis well done and capable of sufficient discrimination to produce confident results, there can also be value in negative results.' 元科学文献(BMC等)佐证: 负结果必须'conclusive, solid, mature, trustworthy ... come from rigorous investigations','hastily-derived and half-baked negative results can be very dangerous.' 对硕士论文的裁决含义: 用冻结评测器/新颖性门控/loop-vs-random对照等安全机制保证严谨性,是让'正确的null'成为可发表贡献的关键——但这也意味着仅凭null本身不足以撑起主贡献,严谨的对照设计才是价值来源。
**源**:https://arxiv.org/html/2406.03980v1

### [7] medium · 综合裁决 (基于claims 4,5,12,13,17的逻辑组合)
诚实裁决与加强建议:'AI4S价值是域相关的——需同时具备可验证评测器和真实headroom;噪声封顶域产出null是正确而非失败'这一核心论点有直接文献支撑;审稿人对'只证null+防overclaim'作主贡献会质疑其足够性,应同时展示框架在有headroom的域能产出正值以证明null是域的问题而非框架的问题。
**证据**:论点的两个支柱各有一手支撑: (a)'需可验证评测器'——AlphaEvolve/FunSearch都把评测器列为先决条件与局限;(b)'需真实headroom'——AlphaEvolve 75%只是match(无headroom则只能null)。防overclaim的价值锚定在AI Scientist独立评估揭示的真实overclaim失败上。加强路径(内含于你的问题、且被文献逻辑支持而非被某篇论文直接证明,故confidence=medium): 在同一框架下并列一个有headroom的域(如你的规划器loop胜随机27%)产出正值 + 目标能耗域产出正确null,构成域内对照(cross-domain contrast),把'框架无能'的替代解释排除,使'null是域性质'的因果主张可辩护。这与R&C-track/负结果论文所要求的'方法学严谨性决定负结果价值'一致。
**源**:https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf ; https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/Mathematical-discoveries-from-program-search-with-large-language-models.pdf ; https://arxiv.org/abs/2406.03980 ; https://arxiv.org/abs/2502.14297

## 待补(强化稳健性)
- 是否存在把AI4S(LLM自动科研)成功用于'噪声真实传感器/无干净ground truth'域并经独立复现确认的任何先例?当前检索仅得到反例缺失,值得再针对机器人/材料/生物传感等具体子域做一轮定向检索以更强地支撑'无先例'主张。
- 硕士论文若以'跨域对照(有headroom域产正值 vs 噪声封顶域产正确null)'为主贡献,在具体答辩委员会/审稿标准下会被打成什么等级?需要该项目所在项目/学校对方法学贡献vs新发现贡献的接受度信息(文献只给了ML社区层面的接受度,未给硕士答辩层面)。
- '噪声封顶在~1.9% ARE'这个上界本身是否需要一个独立的、可辩护的noise-floor估计(如重复测量方差/交叉数据集),才能把'null是域性质'从主张升级为可量化证明?文献支持'严谨性决定负结果价值',但未告诉你噪声地板该如何形式化估计。
- POISE(arXiv 2603.23951)之外,截至2026年是否有更权威(NeurIPS/ICML/Nature系)的、明确处理'噪声真实数据域AI4S会产出null'的实证或位置论文,可替代这个较新且非顶会的引用以增强稳健性?