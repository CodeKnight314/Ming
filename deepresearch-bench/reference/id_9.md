# Reference 9

## Prompt

在计算化学这个领域，我们通常使用Gaussian软件模拟各种情况下分子的结构和性质计算，比如在关键词中加入'field=x+100'代表了在x方向增加了电场。但是，当体系是经典的单原子催化剂时，它属于分子催化剂，在反应环境中分子的朝向是不确定的，那么理论模拟的x方向电场和实际电场是不一致的。请问：通常情况下，理论计算是如何模拟外加电场存在的情况？

## Article

# **计算化学中外加电场模拟：应对分子取向不确定性的策略与进展**

## **1\. 引言**

### **1.1 电场在化学中的普遍影响**

电场，无论是分子内部产生的局域电场还是外部施加的电场，都在化学过程中扮演着至关重要的角色。它们深刻影响分子的结构、电子性质、化学反应活性以及光谱特征 1。例如，外加电场可以引起原子和分子能级的劈裂（斯塔克效应，Stark effect），改变分子的几何构型，影响分子间相互作用，甚至引发或调控化学反应，如电子转移和质子转移 1。在生物化学领域，酶活性位点内由极性氨基酸残基产生的强局域电场被认为是其高效催化能力的关键因素之一 2。在材料科学和表面化学中，电场可以调控表面吸附、分子自组装以及电催化过程 2。

### **1.2 计算模拟面临的挑战：理论与现实的鸿沟**

随着计算化学的发展，理论模拟已成为研究电场效应的重要工具。然而，一个核心挑战在于如何准确地模拟真实化学环境中的电场作用。标准的计算方法，例如在常用的量子化学软件 Gaussian 中使用 Field 关键词，通常施加的是一个均匀的、静态的外加电场（External Electric Field, EEF），并且其方向是相对于输入文件中的分子固定坐标系定义的 7。这种设定与许多实际化学体系（如溶液相、气相或界面）的物理现实存在差距。在这些体系中，分子，特别是像单原子催化剂（Single-Atom Catalysts, SACs）或分子催化剂这样相对较小的实体，由于热运动会经历快速的、随机的或动态的取向变化 1。

这种理论模型与物理现实之间的不匹配构成了模拟电场效应的主要障碍。标准的固定取向模型（例如 Field=X+100）不仅仅是一种实践上的简化，它在概念层面上就与动态体系脱节。计算软件定义的场矢量（X, Y, Z 方向）是相对于为分子几何结构定义的固定笛卡尔坐标系的 7。然而，在真实的液体或气相中，分子（尤其是小的催化剂）由于热能而快速翻滚，相对于任何实验室固定的外加电场，其取向都在不断变化 1。因此，使用 Field=X+100 进行的单次计算仅代表了分子-场相互作用的一个特定的、任意的“快照”。它本身无法预测系综平均性质或动态重取向分子的行为。这就要求发展能够处理取向平均（见 3.1 节）或直接模拟动力学（见 3.3 节）的方法。

### **1.3 聚焦催化领域**

这一挑战在催化领域尤为突出。电场（包括外加电场和局域/内建电场）正日益被视为调控催化活性和选择性的有力工具 1。单原子催化剂（SACs）和分子催化剂因其离散的分子特性和潜在的取向动力学，成为研究电场效应的典型例子 14。准确模拟这些催化剂在电场作用下的行为，对于理解其催化机理和设计更高效的催化体系至关重要。然而，它们的取向不确定性使得标准固定取向模拟的应用受到限制。

### **1.4 报告目标与结构**

本报告旨在全面概述计算化学中模拟外加电场效应的理论方法，重点关注解决分子（特别是催化剂）取向不确定性问题的策略。报告将评估这些策略的适用性，讨论超越均匀静态场的先进技术，并总结当前主流方法的局限性和最佳实践。具体内容安排如下：第二节介绍模拟均匀静态电场的标准方法及其局限性；第三节探讨处理随机分子取向的主要策略，包括取向平均、特定取向选择和分子动力学模拟；第四节聚焦于催化体系中电场效应的模拟，特别是如何处理取向问题；第五节介绍更复杂的电场模拟方法，如局域电场和 QM/MM 方法；第六节讨论结果解释、方法局限性和最佳实践；最后，第七节进行总结并展望未来发展方向。

## **2\. 模拟均匀静态电场：标准方法**

### **2.1 Gaussian 软件中的 Field 关键词**

在量子化学计算中，Gaussian 软件的 Field 关键词是施加均匀静态外加电场（EEF）的标准工具 7。类似的功能也存在于其他主流量子化学软件中，例如 ORCA 20。

* **功能与实现：** Field 关键词允许用户在进行量子化学计算（如单点能、几何优化、频率计算）时，向体系哈密顿量中添加一个代表外加电场的微扰项 7。理论上，对于均匀电场 Eext​, 该微扰项通常为 −μ​⋅Eext​（电偶极相互作用），其中 μ​ 是分子的偶极矩。更准确地说，该关键词修改了单电子哈密顿量 20。它还可以包含更高阶的电多极矩（直至十六极矩）与场的相互作用项 7。  
* **语法：** 常见的语法格式为 Field=M±N。其中 M 指定场的类型和方向，例如 X, Y, Z 代表沿笛卡尔坐标轴的电偶极场，XX, XYZ 等代表更高阶的多极场。± 符号表示场的方向（相对于标准取向或输入方向），N 是一个整数，指定场的强度，大小为 N×0.0001 原子单位（a.u.）7。例如，Field=X+10 表示施加一个沿 X 轴正方向，强度为 0.001 a.u. 的电偶极场。Field=F(M)N 格式则用于施加费米接触微扰 7。所施加的场是相对于分子在输入文件中的初始取向（标准取向或 Z-矩阵定义的取向）而言的 7。  
* **可用性：** Field 关键词可用于多种量子化学方法，包括 Hartree-Fock (HF)、密度泛函理论 (DFT)、组态相互作用 (CI)、耦合簇 (CC) 以及 Møller-Plesset 微扰理论 (MPn) 等，并可用于单点能、几何优化、频率分析、力常数计算和势能面扫描等多种计算类型 8。

### **2.2 Field 关键词的应用场景**

使用 Field 关键词的标准计算可用于：

* 计算分子在电场作用下的性质变化，例如诱导偶极矩、极化率（通过有限场方法计算）、超极化率等响应性质 20。  
* 模拟振动光谱的斯塔克效应（Vibrational Stark Effect, VSE），即外加电场引起的振动频率位移 1。  
* 研究电场对化学反应势垒的影响，评估电场催化的可能性 1。  
* 模拟分子取向被固定或可以合理假设为固定的体系，例如强吸附在特定取向的表面分子，或作为更复杂模型研究的初步探索。

值得注意的是，Field 关键词与计算响应性质（如极化率 α、超极化率 β 等）的有限场（Finite Field, FF）方法密切相关。响应性质定义为能量或偶极矩对电场强度的导数 20。FF 方法通过在零场和一个或多个小的有限场强下（使用 Field 关键词）进行计算，然后通过有限差分来数值求解这些导数 20。因此，Field 关键词是实现 FF 计算的核心功能。这也意味着，使用 Field 关键词通过 FF 方法得到的响应性质的准确性，不仅取决于底层量子化学方法的精度，还受到数值微分本身的限制（如步长选择、数值噪音）23。

### **2.3 标准方法的局限性**

尽管 Field 关键词应用广泛，但其固有的局限性使其难以直接应用于分子取向不确定的体系：

* **固定取向假设：** 最主要的限制是，电场方向是相对于输入的分子几何结构固定的。这与溶液、气相或动态界面中分子不断变化的取向相矛盾（用户查询，1.2 节讨论）。因此，单次计算结果仅代表众多可能取向中的一个，无法直接反映系综平均行为。  
* **对称性问题：** 施加的电场通常会降低分子的对称性。如果在计算中（特别是几何优化或数值导数计算）保留了较高的对称性约束，或者没有使用如 NoSymm 这样的选项来关闭对称性约束，可能会导致计算错误或收敛困难 7。  
* **均匀场假设：** 该方法假设电场在整个分子区域是均匀的，忽略了在某些实验装置（如 STM 针尖）或凝聚相环境（如溶剂化层、酶活性位点）中可能存在的电场梯度。  
* **基组敏感性：** 准确描述分子在电场中的响应（如极化率、超极化率）需要使用足够柔性的基组，通常需要包含弥散函数和极化函数 20。基组不充分可能导致对电场效应的描述不准确。

## **3\. 应对随机分子取向的模拟策略**

当分子的取向不确定时，实验测量的宏观性质通常反映了对所有可能取向的系综平均（对于气相、溶液等各向同性体系）或特定取向分布的平均（对于液晶、表面吸附等各向异性体系）。理论模拟必须设法考虑这种取向平均或分布。

### **3.1 取向平均 (Orientational Averaging)**

* **概念：** 对于各向同性体系，理论上需要计算特定取向下的性质 f(ω)，然后对所有可能的空间取向 ω（通常用三个欧拉角 α,β,γ 参数化）进行积分平均 11。这个平均过程是在旋转群 SO(3) 上进行的。如果不同取向出现的概率不同（例如在部分取向的体系中），则需要引入取向分布函数 P(ω) 作为权重因子。平均值由积分 I=∫f(ω)P(ω)dω 给出 11。在各向同性的流体相中，P(ω) 是常数。  
* **数值实现：** 由于 f(ω) 通常是复杂函数（例如，通过 QM 计算得到的能量或属性），解析积分一般不可行。因此，需要采用数值积分方法，即所谓的“求积”（Quadrature）方法。这涉及选择一组离散的取向点（求积节点）ωi​，计算每个点上的函数值 f(ωi​)，然后通过加权求和 ∑i​wi​f(ωi​) 来近似积分值 11。  
* **求积方法选择：** 选择高效的求积方案至关重要，简单的等间距采样通常效率低下 11。需要根据积分的维度（例如，对于依赖所有三个欧拉角的函数，是在 SO(3) 或等价的四维单位球面 S3 上积分；如果存在对称性，可能简化为在二维球面 S2 上积分）和被积函数的“各向异性”（即函数随取向变化的剧烈程度）来选择合适的求积点集和权重 11。  
* **计算成本：** 取向平均方法计算量巨大，因为它需要对每个求积节点进行一次完整的（通常是 QM）计算。所需的计算次数取决于被积函数的复杂性和期望的精度，可能需要数十到数千次计算 2。  
* **应用实例：** 该方法适用于模拟液体或气体等各向同性体系中的实验测量结果 11。例如，研究 OEEF 对力化学反应中随机取向分子的影响时，需要考虑取向平均 10。

### **3.2 基于特定取向的计算**

* **方法：** 作为完全取向平均的一种简化，研究者有时会选择一个或几个特定的、具有代表性的分子取向进行计算。常见的选择包括：  
  * 将外加电场方向与分子的固有偶极矩矢量对齐 5。  
  * 将电场方向沿着被认为对反应起关键作用的特定化学键或“反应轴”方向 10。例如，在模拟力化学中的键断裂时，将电场沿拉伸方向施加 10；在模拟酶催化或分子催化时，可能沿底物结合或活化的关键方向施加电场 15。  
  * 选择分子在无外场时能量最低的构象进行计算，隐含假设是该构象在系综中占主导地位，或者电场不会显著改变优势构象。  
* **理由：** 这种方法的主要目的是降低计算成本，或者探索电场在特定方向上可能产生的最大或最小效应。  
* **局限性：** 这本质上是一种近似处理。除非所选取的向在系综中占绝对优势，或者所研究的性质对分子取向不敏感，否则计算结果不能代表真实的系综平均值。结果高度依赖于所选取的向。"反应轴"的定义本身也可能存在模糊性（见 4.3 节讨论）。

### **3.3 分子动力学 (MD) 模拟**

* **概念：** MD 模拟通过数值求解牛顿运动方程，追踪体系中每个原子随时间的运动轨迹。它显式地包含了分子的平动、转动和振动。在外加电场存在下进行 MD 模拟，可以自然地捕捉分子相对于电场的动态取向变化过程 5。  
* **电场实现方式：**  
  * **静态场（常规力法 CFM）：** 最常见的方法是在 MD 模拟的每一步，对体系中每个带有部分电荷 qi​ 的原子施加一个额外的、恒定的库仑力 Fi​=qi​Eext​ 29。  
  * **时变场：** 可以通过让 Eext​ 随时间变化来模拟交流电场、脉冲电场等。常用的函数形式包括高斯脉冲 27 或余弦函数 32。这使得研究体系对动态场的响应成为可能 25。  
  * **恒定电势法 (CPM)：** 尤其适用于模拟位于电极之间的体系。该方法保持电极上的电势恒定，允许电极上的电荷根据体系内部的电荷分布进行动态调整。相比 CFM，CPM 能更好地体现体系（如溶剂）对外部电场的屏蔽效应 31。LAMMPS 等 MD 软件实现了 CPM 31。  
* **力场与 QM/MM/ML 方法：**  
  * **经典 MD：** 依赖于经验力场。标准的固定电荷力场无法描述电场诱导的电子极化效应，在强场下可能不准确 26。可极化力场能改善这一点，但更复杂且参数化困难。  
  * **从头算 MD (AIMD)：** 在模拟的每一步使用量子力学（通常是 DFT）计算原子间的相互作用力。能够准确描述电子极化效应，但计算成本极高，严重限制了模拟的时间尺度（通常在皮秒到纳秒量级）和体系大小 26。  
  * **QM/MM MD：** 将体系划分为 QM 区域（如反应中心）和 MM 区域（环境）。电场可以施加到整个体系，或者通过嵌入方案（见第 5 节）考虑 MM 环境对 QM 区的静电作用。这在计算成本和精度之间提供了一种平衡 33。  
  * **机器学习势 (MLP)：** 利用机器学习方法，基于 QM 计算数据训练得到势函数。MLP 可以在接近 QM 精度的同时，将计算速度提高几个数量级 26。一些 MLP 可以直接包含电场作为输入参数进行训练，但需要不同场强下的训练数据 26。扰动神经网络势 (PNNP) MD 等方法则尝试在零场训练数据的基础上，通过微扰理论引入电场效应，避免了对含场训练数据的依赖 26。  
* **结果分析：** 分析 MD 轨迹可以得到体系性质随时间的变化或系综平均值。分子取向通常通过监测分子固有偶极矩与外加电场方向之间的夹角 θ 来表征 27。可以使用取向序参数，如 \<cosθ\> 或 P2​(cosθ)=\<(3cos2θ−1)/2\> 来量化体系的整体取向程度 35。还可以拟合取向随时间的变化，提取取向弛豫时间常数 τ 27。  
* **应用：** MD 模拟广泛应用于研究电场诱导的分子排列 5、电场对分子动力学行为的影响 25、相变（如水的电致冻结 28）、生物膜的电穿孔 30 以及离子在电场下的迁移 31 等。

MD 模拟在捕捉电场诱导取向方面的有效性，关键取决于模拟时间尺度、分子自身重取向时间尺度以及电场特性（强度、频率等）之间的关系。分子的转动扩散时间取决于其尺寸、形状和环境粘度。外场通过与分子偶极矩相互作用产生力矩来驱动取向 27。取向速率与场强、偶极矩大小和转动阻力有关。MD 模拟的时间长度是有限的，尤其是 AIMD 26。如果模拟时间短于在外场下达到平衡取向所需的时间，模拟只能捕捉到瞬态行为 28。反之，有研究表明，即使是蛋白质这样的大分子，在约 0.5 V/nm 的场强下也能在 10 ns 内实现取向，这可能快于强场（如估计的键断裂阈值 \~45 V/nm）引起结构破坏的速度，提示存在一个“先取向后破坏”的时间窗口，使得 MD 可以有效研究取向动力学 27。对于交流电场，频率也很重要：如果场振荡远快于分子的响应时间，取向效应会减弱 36。

此外，需要认识到标准固定电荷力场 MD（使用 CFM 方法）在模拟电场效应时的固有缺陷。固定电荷力场无法描述分子在外场下的电子极化响应——即电子云分布的重新调整 27。同时，在凝聚相中，周围介质（如水分子）也会极化和重排，产生一个“反应场”，部分屏蔽外加电场 31。简单的 CFM 方法（对所有原子施加 F=qE）本身不能很好地描述这种屏蔽效应 31。这可能导致模拟中外场与分子的相互作用被高估。要更准确地捕捉这些效应，需要使用可极化力场、QM/MM、AIMD 或 CPM 等方法 31。不过，一些高级的反应力场（如 ReaxFF，包含电荷平衡方案）在描述极化方面可能优于简单的固定电荷模型 38。

## **4\. 催化体系中的电场效应模拟**

### **4.1 电场作为催化调控手段**

电场可以通过选择性地稳定或去稳定反应物、产物或过渡态来改变化学反应的速率和选择性，特别是对于那些在反应过程中偶极矩发生显著变化的反应（即具有较大的活化偶极矩 Δμ​‡=μ​TS​−μ​Reactant​）1。这一原理不仅适用于均相和多相催化，也被认为是酶催化高效性的重要机制之一 1。

### **4.2 单原子催化剂 (SACs) 与分子催化剂**

* **定义：** SACs 指的是孤立的金属原子分散并锚定在载体材料上形成的催化剂 14。分子催化剂则是指具有催化活性的离散分子实体，通常是金属有机配合物 39。两者都不同于传统的金属纳米颗粒或块状催化剂。  
* **SACs 的电场效应：**  
  * 外加电场可以直接调控孤立金属原子的电子结构（如电荷分布、前线轨道能级）及其与吸附物的相互作用，这种效应被称为“原位静电极化”（onsite electrostatic polarization）14。这提供了一种动态调控 SACs 活性的新途径 14。  
  * 理论研究表明，EEFs 能够调节 SACs 上的吸附能和反应能垒，例如，调控 M1/Fe3S4 上 CO2 的解离 6 或 SACs 的水分解性能 14。  
  * 载体本身也可能受到电场影响，或介导电场对活性位点的影响，并影响 SACs 在电场下的稳定性 6。  
  * SACs 由于其几何结构（例如原子突出载体表面）可能产生局域电场（尖端效应），这也可能对其催化性能产生影响 14。  
* **分子催化剂的电场效应：**  
  * 类似的原理也适用于分子催化剂。电场可以与其偶极矩、极化率相互作用，影响其构象、电子结构以及与底物的相互作用模式 15。  
  * 一个具体的例子是，理论计算发现，沿 Mn-O 轴施加 OEEF 可以显著改变 Mn-corrolazine 催化降解苯并\[a\]芘（BaP）的反应路径，通过改变催化剂活性氧物种的电荷分布和反应性，将原本需要经过有毒环氧化物中间体的三步反应简化为一步直接羟基化反应 15。

### **4.3 催化剂模拟中的取向问题处理**

由于取向平均或完整动力学模拟的复杂性和计算成本，许多针对催化剂的电场效应计算研究采用了简化的取向处理方式：

* **特定轴向对齐：** 一种常见的做法是假设电场方向与某个特定的分子轴或化学键方向一致，例如催化剂与底物之间的关键相互作用轴，或者假定的“反应轴” 10。反应轴通常被定义为过渡态和反应物（或中间体）之间偶极矩的矢量差 (Δμ​‡) 16。例如，15 将电场沿 Mn-O 轴施加；10 将电场沿被拉伸或断裂的化学键方向施加。  
* **代表性方向计算：** 对于负载型催化剂（如 SACs），有时会计算电场平行和垂直于载体表面的几种情况 6。  
* **MD/QM/MM 方法：** 对于需要更真实地考虑环境（如溶剂）和动态效应的场景，MD 或 QM/MM 模拟是更合适的选择。这些方法可以模拟催化剂在溶液中的动态取向，或者模拟由溶剂、蛋白质环境等产生的局域电场（见第 5 节）。例如，4 讨论了使用 QM/MM 模拟酶活性位点电场；46 使用 MD 结合 DFT 研究电极界面附近水的取向及其对表面物种溶剂化能的影响，这与电催化直接相关。  
* **与实验的联系：** 需要注意的是，一些施加电场的实验技术（如扫描隧道显微镜针尖下的分子结 2）本身就可能涉及特定的分子取向。计算模拟有时会尝试模拟这些特定的实验构型。然而，如何将这些特定取向下的发现推广到本体溶液催化仍然是一个挑战。

将电场沿“反应轴”（Δμ​‡）方向施加是简化计算的常用策略 16，但这种做法有其内在的复杂性。首先，反应轴本身可能并非在整个反应路径上保持不变。分子的偶极矩 μ​TS​ 和 μ​Reactant​ 取决于所选的几何结构，它们定义的 Δμ​‡ 只是一个特定方向。然而，真实的反应路径可能涉及分子的旋转或显著的几何变化，这意味着瞬时偶极矩变化最大的方向可能沿着反应坐标是变化的。其次，分子整体具有极化率张量 α。外加电场会诱导一个偶极矩 μ​ind​=α⋅E。这个诱导偶极矩与电场的相互作用能（−21​E⋅α⋅E）取决于电场相对于分子极化率主轴的方向，而这个方向不一定与 Δμ​‡ 一致。因此，仅仅关注 Δμ​‡ 方向可能过度简化了场-分子相互作用，忽略了由极化率主导的效应或电场与分子“旁观”部分的相互作用。正如 15 所指出的，在反应过程中，反应物相对于电场的取向可能发生变化，从而改变了电场效应的大小和方向。

对于单原子催化剂，电场效应可能超越简单的静电相互作用。14 提出的“原位静电极化”机制表明，电场不仅与体系的整体偶极矩相互作用，还能直接、局域地改变活性中心金属原子的前线轨道能量和占据情况。这类似于通过改变配体来调控金属中心的电子性质，但电场提供了一种动态、可逆的调控方式。外场可以直接微扰金属 d 轨道的能级，影响其与吸附物轨道的杂化 14。这种局域电子结构微扰直接关系到 SACs 的成键和反应活性，并被 14 与电荷分布变化和反应决速步骤动力学的改变联系起来。这提示电场可以扮演“虚拟配体”的角色，动态地控制 SAC 活性位点的电子性质（类似于调控氧化态或 d 带中心）。

## **5\. 先进与替代的电场模拟方法**

除了标准的均匀静态场模拟和应对取向问题的策略外，还发展了更先进和替代的方法来模拟更复杂的电场效应或环境。

### **5.1 超越均匀静态场**

* **时变电场：** 如 3.3 节所述，MD 模拟可以方便地引入随时间变化的电场 E(t)，例如模拟交流电场（正弦波）、脉冲电场（高斯型或其他形状）等 25。这使得研究体系的动态响应，如介电弛豫 45、动态取向过程 27 以及特定频率（如太赫兹、吉赫兹）电场的影响成为可能 36。虽然原则上也可以在含时量子力学方法中引入时变场，但这在实践中不如 MD 普遍。  
* **电场梯度：** 标准方法通常忽略电场梯度。模拟非均匀电场更为复杂，缺乏标准化流程，但对于模拟某些实验装置（如针尖电极）或高度局域化的现象可能具有重要意义。虽然一些 QM 软件可以计算原子核位置处的电场梯度，但这通常是作为一种分析属性，而不是作为施加的外部条件 21。

### **5.2 模拟局域电场**

* **来源：** 在凝聚相或复杂环境中（如酶活性位点、溶剂笼、表面/电极附近），目标分子会受到周围环境（溶剂分子、离子、蛋白质残基、表面原子等）产生的局域电场作用。这些局域电场可以非常强且高度有序，并在决定分子行为和功能（如酶催化 2、溶剂化效应 46）中扮演关键角色。  
* **隐式溶剂模型 (PCM)：** 像极化连续介质模型 (PCM) 这样的方法，通过将溶剂视为可极化的连续介质来近似其平均反应场，但无法描述由特定溶剂分子或环境基团产生的结构化、非均匀的局域电场 39。  
* **显式环境模型 (MD/QM)：** 通过 MD 模拟（经典或 QM/MM）显式地包含环境分子（溶剂、蛋白质等）。可以直接计算由环境部分（MM 部分）在目标分子（QM 部分）区域产生的静电势或电场 46。例如，47 计算了水分子在石墨烯纳米带边缘产生的局域电场；46 模拟了电极界面附近水产生的电场；33 利用 QM/MM MD 快照计算了酮类固醇异构酶（KSI）活性位点的电场。

### **5.3 混合量子力学/分子力学 (QM/MM) 方法**

* **概念：** QM/MM 方法将整个体系划分为两个区域：需要高精度量子力学描述的核心区域（QM 区域，如反应物分子、酶活性位点）和可以用经典力场描述的环境区域（MM 区域）33。这种划分使得在高精度水平上研究大体系中的局域化学过程成为可能 49。  
* **模拟局域电场：** MM 环境通过静电相互作用对 QM 区域产生影响，这实际上就模拟了环境产生的局域电场。QM/MM 之间的耦合通过不同的“嵌入”（Embedding）方案实现：  
  * **力学嵌入 (Mechanical Embedding)：** 最简单，仅考虑 QM/MM 边界处的空间位阻（如范德华作用），不包含静电耦合。对于极性体系，其精度通常不足 50。  
  * **静电嵌入 (Electrostatic Embedding)：** 将 MM 原子的部分电荷包含在 QM 计算的哈密顿量中。QM 区域的电子密度会受到 MM 电荷产生的静态电场的极化 33。这是目前最常用的嵌入方案。  
  * **极化嵌入 (Polarizable Embedding)：** 不仅考虑 MM 对 QM 的静电作用，还允许 MM 环境的电子分布响应 QM 区域的电荷变化（反之亦然，通常需要自洽迭代）。这需要使用可极化力场或特殊方法（如直接反应场 DRF 53）。这种方案更物理真实，但也更耗费计算资源。它对于准确描述不同电子态（如基态 vs 激发态）的微分溶剂化或极化效应至关重要 53。  
* **边界处理：** 需要特殊技术来处理跨越 QM/MM 边界的化学键，例如使用“连接原子”（Link Atoms）或专门的边界处理方案（如 GMFCC 33、RCD 52）。  
* **应用：** QM/MM 方法被广泛应用于模拟酶催化 4、溶液相反应 34、表面化学（如沸石催化 50）以及复杂环境中的光化学过程 53。结合 MD 进行 QM/MM 模拟，可以动态地采样构象并计算电场和相关性质 33。

QM/MM 中静电嵌入和极化嵌入的选择，不仅仅是定量精度的问题，有时会影响对涉及显著电荷重新分布或电子激发过程的定性描述。静电嵌入能捕捉环境的静态电场，但忽略了环境对 QM 区域变化的电子响应 53。在化学反应或电子激发过程中，QM 区域的电荷分布会改变。可极化的环境会对这种变化做出快速的电子响应（电子极化）和可能的较慢的原子核重排响应 53。静电嵌入无法描述这种快速的电子响应，这对于准确计算不同状态间的溶剂化能差异（例如，基态和激发态之间的溶剂化致变色位移 53）或稳定电荷分离的中间体可能至关重要。极化嵌入方案（如 DRF 53）明确地模拟了这种电子响应，提供了更完整的物理图像，尽管计算成本更高。53 强调了其对激发态的重要性。34 也指出，即使是拟合的 ML/MM 势也需要考虑极化的 QM 粒子与 MM 粒子的相互作用。

### **5.4 机器学习方法**

* **机器学习势 (MLP)：** 基于 QM 数据训练的 MLP 可以预测能量和力，用于进行 MD 模拟，能够在较低的计算成本下达到接近 QM 的精度 26。  
* **含场 MLP：** 一些 MLP 在训练时就包含了外加电场作为特征，可以直接预测分子在电场下的性质 26。但这种方法需要针对不同场强进行大量的训练数据 26。  
* **微扰 ML：** 像 PNNP MD 这样的方法，仅使用零场 QM 数据训练 MLP 来预测零场势能和偶极矩/原子极化张量 (APT)，然后将电场相互作用作为微扰项加入 26。这避免了对大量含场训练数据的需求。  
* **潜力：** 机器学习为在可接受的时间尺度内高精度模拟大体系在外场下的行为提供了一条有前景的途径。

### **5.5 专用软件/工作流**

为了简化含场计算的流程，一些专门的工具和工作流被开发出来。例如，A.V.E.D.A. (Automated Variable Electric-Field DFT Application) 软件可以自动化地在沿最优方向（通常是 Δμ​‡ 轴）且强度递增的 OEEF 中优化反应物和过渡态结构 16。这类工具降低了进行含场计算的技术门槛，使得非计算专家也能更容易地评估反应对电场效应的敏感性。

先进的模拟方法（如 MD、QM/MM）能够计算分子感受到的局域电场，这与利用振动斯塔克效应 (VSE) 作为探针的实验研究形成了强大的协同作用。VSE 指的是特定化学键（如 C=O, C-D, C≡N）的振动频率会随着投射到该键轴上的局域电场强度发生线性变化 3。实验上可以通过测量探针分子在不同环境（溶剂、酶活性位点）中的振动频率位移来探测局域电场 3。计算模拟则可以在原子尺度上计算探针分子在相同环境中所经历的涨落电场 3。通过比较模拟得到的电场与实验测得的频率位移，研究人员不仅可以验证模拟方法的可靠性，还能获得关于局域电场大小和方向的详细信息 3。3 的研究利用一个同时具有 C=O 和 C-D 键的双向探针，结合模拟，成功地绘制了溶剂和酶活性位点中电场的方向，并发现了两者之间存在显著差异。这种计算与实验的结合为理解分子尺度的静电环境提供了强有力的手段。

## **6\. 结果解释、局限性与最佳实践**

在模拟外加电场效应时，选择合适的计算方法并正确解释结果至关重要。以下总结了不同方法的特点、局限性以及应用中的注意事项。

### **6.1 方法总结与权衡**

**表 1：模拟外加电场效应主要方法的比较概述**

| 方法 | 描述 | 取向处理 | 极化效应捕捉 | 动态性捕捉 | 优点 | 局限性 | 典型应用 | 关键文献示例 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 标准静态场 (如 Gaussian Field) | 在 QM 计算中施加均匀静态场，分子取向固定 | 固定 | QM 方法本身 | 否 | 实现简单，计算成本相对较低（单次计算） | 无法处理随机取向；忽略屏蔽；均匀场假设；对称性问题；基组敏感性 | 计算响应性质 (FF)；固定取向体系模拟；初步探索 | 7 |
| 取向平均 | 对多个离散取向的 QM 计算结果进行数值积分平均 | 平均 | QM 方法本身 | 否 (系综平均) | 适用于各向同性体系的系综平均性质 | 计算成本极高（多次 QM 计算）；求积方法选择和收敛性问题 | 模拟气相/液相实验；计算系综平均光谱/能量 | 10 |
| MD (经典力场) | 使用经典力场模拟分子在外场下的运动轨迹 | 动态 | 否 (固定电荷)\<sup\>a\</sup\> | 是 | 计算速度快，可模拟大体系和长时间尺度；自然包含动力学和取向变化 | 力场精度有限，尤其对极化描述不足；可能高估场效应（CFM）；需要长时间模拟以充分采样 | 模拟分子取向/排列；电穿孔；离子输运；宏观体系动力学 | 5 |
| MD (AIMD/QM/MM/ML) | 使用 QM、QM/MM 或 ML 势计算力，进行 MD 模拟 | 动态 | 是 | 是 | 精度高（尤其 AIMD/ML）；能描述电子极化和化学反应；QM/MM/ML 平衡了精度与效率 | 计算成本高（AIMD \> QM/MM \> ML \> 经典）；模拟时间和体系大小受限（尤其 AIMD）；QM/MM 划分和边界处理；ML 依赖训练数据 | 模拟含场反应动力学；溶液/界面/生物体系中的场效应；需要高精度描述电子结构的动态过程 | 25 |
| QM/MM (局域场) | QM/MM 计算，关注 MM 环境对 QM 区域产生的局域电场 | 静态/动态\<sup\>b\</sup\> | 是 (嵌入) | 静态/动态\<sup\>b\</sup\> | 能模拟复杂环境（蛋白、溶剂）产生的局域场；平衡精度与成本 | 结果依赖 QM/MM 划分、嵌入方案和边界处理；力场精度影响 MM 区贡献 | 酶催化；溶液相反应；光化学；模拟 VSE 实验 | 4 |

\<sup\>a\</sup\> 可极化力场可以部分捕捉极化效应。  
\<sup\>b\</sup\> 取决于是否结合 MD 进行模拟。

### **6.2 模拟结果的解释**

* **固定取向计算：** 结果仅代表所选特定取向下的情况。可用于判断电场在特定方向上 *可能* 产生的影响，或比较不同方向效应的相对大小，但不能直接用于预测系综平均值。解释结果时需谨慎，避免过度推广。  
* **取向平均计算：** 结果代表了各向同性体系的系综平均性质。其准确性取决于所用求积方法的质量和采样点数是否足够收敛。  
* **MD 模拟：** 提供动态信息和时间平均性质。结果的可靠性取决于力场或 QM 方法的精度（特别是对极化的描述 26）、模拟时长（是否达到平衡或充分采样 28）以及电场的施加方式（CFM vs CPM 31）。需要分析取向分布函数或序参数来理解分子的排列情况 27。  
* **QM/MM 模拟：** 结果强烈依赖于 QM/MM 的划分方式、嵌入方案的选择（力学、静电、极化）50 以及边界处理技术 52。它提供了关于局域环境效应的深入见解。

### **6.3 关键局限性与考量**

* **计算成本：** 基于 QM 的方法（取向平均、AIMD、QM/MM）计算成本高昂，限制了可研究的体系大小和模拟时间尺度 2。经典 MD 速度快但电子结构描述精度低。ML 方法试图在两者间取得平衡 26。  
* **力场精度：** 经典 MD 的结果受限于力场描述分子间相互作用和分子内性质（尤其是在电场下的极化响应）的准确性 26。  
* **DFT 在电场下的精度：** DFT 的性能可能随泛函和基组的选择而变化，尤其是在强电场下，误差可能显著增加 23。对于关键应用，可能需要进行基准测试或与更高精度方法（如耦合簇）进行比较 23。范围分离泛函（如 ωB97X-V）通常被认为在含场计算中表现较好 10。  
* **采样问题：** 在 MD 或取向平均方法中，获得充分的构象和取向采样可能是一个挑战，需要足够长的模拟时间或足够多的采样点 11。  
* **场强问题：** 为了在可行的模拟时间内观察到显著效应，计算模拟中使用的电场强度往往远高于宏观实验中可实现的场强 28。将模拟结果外推到较低场强时需要谨慎，不能简单假设线性响应关系始终成立 37。强场也可能导致 DFT 计算精度下降 23。  
* **取向/对齐的定义：** 如何定义和测量分子的取向（例如，依据偶极矩矢量、特定化学键轴、惯性主轴等）会影响结果的解释 27。

### **6.4 最佳实践建议**

* **明确研究问题：** 首先确定研究目标：是需要系综平均性质，还是关心特定方向的效应？动力学过程是否重要？  
* **选择合适方法：** 根据研究问题、体系特性以及计算资源，权衡精度与成本，选择最合适的方法。  
* **谨慎使用固定取向：** 如果采用固定取向计算，需明确说明理由并承认其局限性。如果可行，计算多个代表性取向进行比较。  
* **确保平均收敛：** 如果进行取向平均，应使用合适的数值积分方法，并检验结果随采样点数增加的收敛性 11。  
* **细致进行 MD 模拟：** 如果使用 MD，需仔细选择力场或 QM/MM 设置，确保模拟时间足够长，并明确所用的电场施加方法 31。显式分析分子的取向行为 27。  
* **选用合适基组：** 对于涉及电场的 QM 计算，应使用包含弥散和极化函数的基组，以准确描述电子云的响应 20。  
* **关注 DFT 局限：** 认识到 DFT 在强场下可能存在的精度问题 23，必要时进行验证。  
* **清晰报告细节：** 在发表结果时，应详细报告所有模拟参数和方法选择，并提供选择这些参数的理由。

**表 2：常见计算化学软件中电场模拟功能的实现（部分示例）**

| 软件 | 关键词/方法 | 场类型 | 取向/实现说明 | 关键文献示例 |
| :---- | :---- | :---- | :---- | :---- |
| Gaussian | Field | 均匀静态电多极场 (至十六极)；费米接触微扰 | 相对于输入坐标系的固定取向；通过修改单电子哈密顿量实现 | 7 |
| Gaussian | External | 外部程序接口 | 可用于 ONIOM 计算或调用外部程序计算能量/力，理论上可结合外部程序模拟特殊场，但主要用于 QM/MM 或接口 | 54 |
| ORCA | EField (在 %method 块中) | 均匀静态电偶极场 | 类似 Gaussian Field，指定场矢量 (X, Y, Z 分量) | 20 |
| Q-Chem | electric\_field (在 $rem 中) | 均匀静态电偶极场 | 指定场矢量 (X, Y, Z 分量)；也用于 EFEI 方法施加拉伸力 | 10 |
| LAMMPS | fix efield | 均匀静态或时变电场 (CFM) | 对带电原子施加力 F=qE(t) | 29 |
| LAMMPS | fix electrode/conp 或类似 (CPM) | 恒定电极电势 | 模拟电极间的体系，电极电荷动态调整以维持恒定电势，能体现屏蔽效应 | 31 |
| GROMACS | electric-field-x/y/z (在 .mdp 文件中) | 均匀静态或周期性时变 (余弦) 电场 (CFM) | 对带电原子施加力；时变场可指定频率、幅度和相位 | 32 |
| NAMD | QM/MM 模块 (与外部 QC 软件接口) | QM/MM 嵌入场；可施加外部场 | MM 环境对 QM 区产生静电嵌入场；可额外施加均匀场；支持 link atom 等边界处理 | 52 |
| CPMD | QM/MM (通过 MiMiC 与 GROMACS 耦合) | QM/MM 嵌入场；可施加外部场 | 类似 NAMD，MM 环境提供嵌入场；MiMiCPy 工具用于简化输入文件准备 | 33 |

注意：此表仅为示例，具体功能和实现可能随软件版本更新而变化。

## **7\. 结论与展望**

### **7.1 总结**

模拟外加电场对分子体系的影响是计算化学中的一个重要研究方向，尤其在催化、材料科学和生物化学领域具有广泛应用。然而，标准计算方法（如 Gaussian 的 Field 关键词）通常施加一个相对于分子固定坐标系的均匀静态场，这与许多实际体系中分子取向随机或动态变化的现实不符。这一“取向问题”是准确模拟电场效应的核心挑战。

为了应对这一挑战，研究人员发展了多种策略：

1. **取向平均：** 通过对大量离散取向的计算结果进行数值积分，得到系综平均性质。适用于各向同性体系，但计算成本极高。  
2. **特定取向计算：** 选择一个或几个代表性取向进行计算，作为一种简化。结果依赖于取向的选择，不能完全代表系综行为。  
3. **分子动力学 (MD)：** 显式模拟分子的动态演化，自然地包含了取向变化。结合不同的力场（经典、可极化、ML）或 QM/MM 方法，可以在不同精度和成本水平上进行模拟。但结果受限于力场/模型的准确性、模拟时间和采样充分性。  
4. **QM/MM 方法：** 特别适用于模拟大体系中的局域化学过程和环境（如溶剂、蛋白质）产生的局域电场。嵌入方案的选择（静电 vs. 极化）对结果有重要影响。

### **7.2 核心信息**

模拟电场效应时，不存在一种“万能”的方法。方法的选择必须基于具体的研究问题、目标体系的特性、所需精度以及可用的计算资源。深刻理解所选方法的内在假设、优势和局限性，对于可靠地解释模拟结果至关重要。对于分子取向不确定的体系，简单地使用固定取向的标准静态场计算得出的结论需要特别谨慎。

### **7.3 催化领域的意义**

在催化领域，无论是分子催化剂、单原子催化剂还是酶催化，电场都被证明是影响其活性和选择性的关键因素。准确模拟电场效应，特别是解决取向不确定性问题，对于理解催化机理、预测催化性能和理性设计新型高效催化剂具有重要意义。尽管面临挑战，但结合先进的模拟策略（如 MD、QM/MM、取向平均）和理论分析（如“原位静电极化”机制），计算化学正为揭示电场在催化中的作用提供越来越强大的工具。

### **7.4 未来方向与展望**

未来，计算化学模拟电场效应的研究有望在以下几个方面取得进展：

* **更精确、高效的模型：** 发展更准确且计算效率更高的可极化力场和 QM/MM 嵌入方案 28，以及更可靠、泛化能力更强的机器学习势，特别是那些能有效处理电场效应的 MLP 26。  
* **高效的取向平均算法：** 开发和应用更先进的数值积分技术，以更少的采样点实现对复杂高维取向空间的精确平均 11。  
* **模拟与实验的紧密结合：** 加强计算模拟与实验（尤其是利用 VSE 等光谱探针技术 3）的协同，通过相互验证和补充，深入理解分子尺度的电场环境及其作用。  
* **易用性工具的开发：** 开发更多用户友好的软件和工作流（如 A.V.E.D.A. 16），降低进行复杂含场模拟的技术门槛，促进其在更广泛领域的应用。  
* **应对弱场和长时间尺度挑战：** 发展能够有效模拟低频、低强度电场效应的方法，克服当前模拟在时间尺度上的限制，以更接近真实的实验条件 36。

随着计算能力的提升和理论方法的发展，计算化学将在揭示和利用电场调控化学过程方面发挥日益重要的作用。

#### **引用的著作**

1. Effects and Influence of External Electric Fields on the Equilibrium Properties of Tautomeric Molecules, 访问时间为 四月 24, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC9865840/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9865840/)
2. Fast and Simple Evaluation of the Catalysis and Selectivity Induced by External Electric Fields \- ACS Publications, 访问时间为 四月 24, 2025， [https://pubs.acs.org/doi/10.1021/acscatal.1c04247](https://pubs.acs.org/doi/10.1021/acscatal.1c04247)
3. A two-directional vibrational probe reveals different electric field orientations in solution and an enzyme active site, 访问时间为 四月 24, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC10082611/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10082611/)
4. From random to rational: improving enzyme design through electric fields, second coordination sphere interactions, and conformational dynamics \- RSC Publishing Home, 访问时间为 四月 24, 2025， [https://pubs.rsc.org/en/content/articlehtml/2023/sc/d3sc02982d](https://pubs.rsc.org/en/content/articlehtml/2023/sc/d3sc02982d)
5. Manipulating molecular orientation in vapor-deposited organic semiconductor glasses via in situ electric fields \- Digital CSIC, 访问时间为 四月 24, 2025， [https://digital.csic.es/bitstream/10261/382581/1/manipulating\_molecular\_orientation.pdf](https://digital.csic.es/bitstream/10261/382581/1/manipulating_molecular_orientation.pdf)
6. Single-atom catalysis for carbon dioxide dissociation using greigite \- White Rose Research Online, 访问时间为 四月 24, 2025， [https://eprints.whiterose.ac.uk/199486/8/Single-atom%20catalysis%20for%20carbon%20dioxide%20dissociation%20using%20greigitesupported.pdf](https://eprints.whiterose.ac.uk/199486/8/Single-atom%20catalysis%20for%20carbon%20dioxide%20dissociation%20using%20greigitesupported.pdf)
7. Field \- G09 Keyword, 访问时间为 四月 24, 2025， [http://wild.life.nctu.edu.tw/\~jsyu/compchem/g09/g09ur/k\_field.htm](http://wild.life.nctu.edu.tw/~jsyu/compchem/g09/g09ur/k_field.htm)
8. Field | Gaussian.com, 访问时间为 四月 24, 2025， [https://gaussian.com/field/](https://gaussian.com/field/)
9. Re: CCL:How to study the effect of external electric field using Gaussian, 访问时间为 四月 24, 2025， [https://server.ccl.net/chemistry/resources/messages/2000/09/13.018-dir/](https://server.ccl.net/chemistry/resources/messages/2000/09/13.018-dir/)
10. Using Oriented External Electric Fields to Manipulate Rupture Forces of Mechanophores \- ChemRxiv, 访问时间为 四月 24, 2025， [https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/645104001ca6101a45a93741/original/using-oriented-external-electric-fields-to-manipulate-rupture-forces-of-mechanophores.pdf](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/645104001ca6101a45a93741/original/using-oriented-external-electric-fields-to-manipulate-rupture-forces-of-mechanophores.pdf)
11. Numerical evaluation of orientation averages and its application to molecular physics, 访问时间为 四月 24, 2025， [https://pubs.aip.org/aip/jcp/article/161/13/131501/3315373/Numerical-evaluation-of-orientation-averages-and](https://pubs.aip.org/aip/jcp/article/161/13/131501/3315373/Numerical-evaluation-of-orientation-averages-and)
12. Numerical evaluation of orientation averages and its application to molecular physics \- AIP Publishing, 访问时间为 四月 24, 2025， [https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/5.0230569/20193075/131501\_1\_5.0230569.pdf](https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/5.0230569/20193075/131501_1_5.0230569.pdf)
13. Orientation of Chiral Molecules by External Electric Fields: Focus on Photodissociation Dynamics \- MDPI, 访问时间为 四月 24, 2025， [https://www.mdpi.com/2073-8994/14/10/2152](https://www.mdpi.com/2073-8994/14/10/2152)
14. Boosting the performance of single-atom catalysts via external electric field polarization, 访问时间为 四月 24, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC9163078/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9163078/)
15. (PDF) The High-Effective Catalytic Degradation of Benzo\[a\]pyrene by Mn-Corrolazine Regulated by Oriented External Electric Field: Insight From DFT Study \- ResearchGate, 访问时间为 四月 24, 2025， [https://www.researchgate.net/publication/361066988\_The\_High-Effective\_Catalytic\_Degradation\_of\_Benzoapyrene\_by\_Mn-Corrolazine\_Regulated\_by\_Oriented\_External\_Electric\_Field\_Insight\_From\_DFT\_Study](https://www.researchgate.net/publication/361066988_The_High-Effective_Catalytic_Degradation_of_Benzoapyrene_by_Mn-Corrolazine_Regulated_by_Oriented_External_Electric_Field_Insight_From_DFT_Study)
16. Automated Variable Electric-Field DFT Application for Evaluation of Optimally Oriented Electric Fields on Chemical Reactivity, 访问时间为 四月 24, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC9830642/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9830642/)
17. Three-Dimensional Activity Volcano Plot under an External Electric Field, 访问时间为 四月 24, 2025， [https://liutheory.westlake.edu.cn/pdf/acscatal.2c04961.pdf](https://liutheory.westlake.edu.cn/pdf/acscatal.2c04961.pdf)
18. Challenges and Opportunities in Engineering the Electronic Structure of Single-Atom Catalysts | ACS Catalysis \- ACS Publications, 访问时间为 四月 24, 2025， [https://pubs.acs.org/doi/10.1021/acscatal.2c05992](https://pubs.acs.org/doi/10.1021/acscatal.2c05992)
19. Challenges and Opportunities in Engineering the Electronic Structure of Single-Atom Catalysts \- PMC \- PubMed Central, 访问时间为 四月 24, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC9990067/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9990067/)
20. Quantum chemistry in external electrostatic field? \- Matter Modeling Stack Exchange, 访问时间为 四月 24, 2025， [https://mattermodeling.stackexchange.com/questions/6714/quantum-chemistry-in-external-electrostatic-field](https://mattermodeling.stackexchange.com/questions/6714/quantum-chemistry-in-external-electrostatic-field)
21. Prop \- Gaussian.com, 访问时间为 四月 24, 2025， [https://gaussian.com/prop/](https://gaussian.com/prop/)
22. Automated Variable Electric-Field DFT Application for Evaluation of Optimally Oriented Electric Fields on Chemical Reactivity | The Journal of Organic Chemistry \- ACS Publications, 访问时间为 四月 24, 2025， [https://pubs.acs.org/doi/10.1021/acs.joc.2c01893](https://pubs.acs.org/doi/10.1021/acs.joc.2c01893)
23. (PDF) How Accurate is Density Functional Theory for Molecules in Electric Fields?, 访问时间为 四月 24, 2025， [https://www.researchgate.net/publication/360446608\_How\_Accurate\_is\_Density\_Functional\_Theory\_for\_Molecules\_in\_Electric\_Fields](https://www.researchgate.net/publication/360446608_How_Accurate_is_Density_Functional_Theory_for_Molecules_in_Electric_Fields)
24. Numerical evaluation of orientation averages and its application to molecular physics \- arXiv, 访问时间为 四月 24, 2025， [https://arxiv.org/html/2407.17434v1](https://arxiv.org/html/2407.17434v1)
25. Effects of Externally Applied Electric Fields on the Manipulation of Solvated-Chignolin Folding: Static- versus Alternating-Field Dichotomy at Play | The Journal of Physical Chemistry B \- ACS Publications, 访问时间为 四月 24, 2025， [https://pubs.acs.org/doi/10.1021/acs.jpcb.1c06857](https://pubs.acs.org/doi/10.1021/acs.jpcb.1c06857)
26. Molecular dynamics simulation with finite electric fields using Perturbed Neural Network Potentials \- arXiv, 访问时间为 四月 24, 2025， [https://arxiv.org/html/2403.12319v1](https://arxiv.org/html/2403.12319v1)
27. Protein orientation in time-dependent electric fields: orientation ..., 访问时间为 四月 24, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC8456286/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8456286/)
28. Molecular Simulation in External Electric and Electromagnetic Fields \- CECAM, 访问时间为 四月 24, 2025， [https://www.cecam.org/workshop-details/molecular-simulation-in-external-electric-and-electromagnetic-fields-783](https://www.cecam.org/workshop-details/molecular-simulation-in-external-electric-and-electromagnetic-fields-783)
29. Manipulating molecular orientation in vapor-deposited organic ..., 访问时间为 四月 24, 2025， [https://pubs.rsc.org/en/content/articlehtml/2024/tc/d4tc03271c](https://pubs.rsc.org/en/content/articlehtml/2024/tc/d4tc03271c)
30. Membrane Electroporation: A Molecular Dynamics Simulation \- PMC, 访问时间为 四月 24, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC1305635/](https://pmc.ncbi.nlm.nih.gov/articles/PMC1305635/)
31. Comparative study of external electric field and potential effects on liquid water ions, 访问时间为 四月 24, 2025， [https://www.tandfonline.com/doi/full/10.1080/00268976.2021.1998689](https://www.tandfonline.com/doi/full/10.1080/00268976.2021.1998689)
32. How should I set up the parameters of the external electric field in mdp file? \- ResearchGate, 访问时间为 四月 24, 2025， [https://www.researchgate.net/post/How\_should\_I\_set\_up\_the\_parameters\_of\_the\_external\_electric\_field\_in\_mdp\_file](https://www.researchgate.net/post/How_should_I_set_up_the_parameters_of_the_external_electric_field_in_mdp_file)
33. An Ab Initio QM/MM Study of the Electrostatic Contribution to ..., 访问时间为 四月 24, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC6222312/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6222312/)
34. Machine Learning in QM/MM Molecular Dynamics Simulations of Condensed-Phase Systems | Journal of Chemical Theory and Computation \- ACS Publications, 访问时间为 四月 24, 2025， [https://pubs.acs.org/doi/10.1021/acs.jctc.0c01112](https://pubs.acs.org/doi/10.1021/acs.jctc.0c01112)
35. Analyzing MD trajectory: molecule orientation \- Matter Modeling Stack Exchange, 访问时间为 四月 24, 2025， [https://mattermodeling.stackexchange.com/questions/4580/analyzing-md-trajectory-molecule-orientation](https://mattermodeling.stackexchange.com/questions/4580/analyzing-md-trajectory-molecule-orientation)
36. Perspectives on external electric fields in molecular simulation: Progress, prospects and challenges | Request PDF \- ResearchGate, 访问时间为 四月 24, 2025， [https://www.researchgate.net/publication/275359999\_Perspectives\_on\_external\_electric\_fields\_in\_molecular\_simulation\_Progress\_prospects\_and\_challenges](https://www.researchgate.net/publication/275359999_Perspectives_on_external_electric_fields_in_molecular_simulation_Progress_prospects_and_challenges)
37. Reaction Rate Theory for Electric Field Catalysis in Solution \- arXiv, 访问时间为 四月 24, 2025， [https://arxiv.org/html/2404.01455v1](https://arxiv.org/html/2404.01455v1)
38. Atomistic insight into the effects of electrostatic fields on hydrocarbon reaction kinetics, 访问时间为 四月 24, 2025， [https://pubs.aip.org/aip/jcp/article/158/5/054109/2871534/Atomistic-insight-into-the-effects-of](https://pubs.aip.org/aip/jcp/article/158/5/054109/2871534/Atomistic-insight-into-the-effects-of)
39. Software for the frontiers of quantum chemistry: An overview of developments in the Q-Chem 5 package \- PubMed Central, 访问时间为 四月 24, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC9984241/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9984241/)
40. Dye-sensitized solar cells strike back \- PMC \- PubMed Central, 访问时间为 四月 24, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC8591630/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8591630/)
41. Publications \- RWTH AACHEN UNIVERSITY Department of Chemistry \- English, 访问时间为 四月 24, 2025， [https://www.chemie.rwth-aachen.de/cms/chemie/forschung/forschungsschwerpunkte/\~mdezi/publikationen/?mobile=1\&lidx=1](https://www.chemie.rwth-aachen.de/cms/chemie/forschung/forschungsschwerpunkte/~mdezi/publikationen/?mobile=1&lidx=1)
42. Publications of ITMC \- RWTH AACHEN UNIVERSITY ITMC \- English, 访问时间为 四月 24, 2025， [https://www.itmc.rwth-aachen.de/cms/itmc/forschung/\~kdbk/publikationen-des-itmc/?mobile=1\&lidx=1](https://www.itmc.rwth-aachen.de/cms/itmc/forschung/~kdbk/publikationen-des-itmc/?mobile=1&lidx=1)
43. High-Throughput Screening Approach for Catalytic Applications through Regulation of Adsorption Energies via Work Function | Langmuir \- ACS Publications, 访问时间为 四月 24, 2025， [https://pubs.acs.org/doi/10.1021/acs.langmuir.4c03385](https://pubs.acs.org/doi/10.1021/acs.langmuir.4c03385)
44. Book of Abstracts | CESTC 2019, 访问时间为 四月 24, 2025， [https://cestc2019.univie.ac.at/wp-content/uploads/book-of-abstracts/cestc2019\_book-of-abstracts.pdf](https://cestc2019.univie.ac.at/wp-content/uploads/book-of-abstracts/cestc2019_book-of-abstracts.pdf)
45. Orientation Polarization Spectroscopy—Toward an Atomistic Understanding of Dielectric Relaxation Processes \- PMC, 访问时间为 四月 24, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC9330800/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9330800/)
46. Influence of an Electrified Interface on the Entropy and Energy of Solvation of Methanol Oxidation Intermediates on Platinum(111, 访问时间为 四月 24, 2025， [https://pubs.rsc.org/en/content/getauthorversionpdf/d1cp05358b](https://pubs.rsc.org/en/content/getauthorversionpdf/d1cp05358b)
47. QM/MM computed local electric field produced by the water molecules for... | Download Scientific Diagram \- ResearchGate, 访问时间为 四月 24, 2025， [https://www.researchgate.net/figure/QM-MM-computed-local-electric-field-produced-by-the-water-molecules-for-two-different\_fig6\_231646495](https://www.researchgate.net/figure/QM-MM-computed-local-electric-field-produced-by-the-water-molecules-for-two-different_fig6_231646495)
48. Testing the Limitations of MD-based Local Electric Fields Using the Vibrational Stark Effect in Solution \- ChemRxiv, 访问时间为 四月 24, 2025， [https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/60c7542ebb8c1ac3353dc1be/original/testing-the-limitations-of-md-based-local-electric-fields-using-the-vibrational-stark-effect-in-solution-penicillin-g-as-a-test-case.pdf](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/60c7542ebb8c1ac3353dc1be/original/testing-the-limitations-of-md-based-local-electric-fields-using-the-vibrational-stark-effect-in-solution-penicillin-g-as-a-test-case.pdf)
49. Introduction to QM/MM Simulations \- MPG.PuRe, 访问时间为 四月 24, 2025， [https://pure.mpg.de/pubman/item/item\_1562917\_8/component/file\_1744080/1562917.pdf](https://pure.mpg.de/pubman/item/item_1562917_8/component/file_1744080/1562917.pdf)
50. The application of QM/MM simulations in heterogeneous catalysis ..., 访问时间为 四月 24, 2025， [https://pubs.rsc.org/en/content/articlehtml/2023/cp/d2cp04537k](https://pubs.rsc.org/en/content/articlehtml/2023/cp/d2cp04537k)
51. MiMiCPy: An Efficient Toolkit for MiMiC-Based QM/MM Simulations \- PMC \- PubMed Central, 访问时间为 四月 24, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC10015468/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10015468/)
52. Hybrid QM/MM NAMD, 访问时间为 四月 24, 2025， [http://www.ks.uiuc.edu/Research/qmmm/](http://www.ks.uiuc.edu/Research/qmmm/)
53. Multistate, Polarizable QM/MM Embedding Scheme Based on the Direct Reaction Field Method: Solvatochromic Shifts, Analytical Gradients and Optimizations of Conical Intersections in Solution \- ACS Publications, 访问时间为 四月 24, 2025， [https://pubs.acs.org/doi/10.1021/acs.jctc.3c01018](https://pubs.acs.org/doi/10.1021/acs.jctc.3c01018)
54. G09 Keyword: External, 访问时间为 四月 24, 2025， [http://sobereva.com/g09/k\_external.htm](http://sobereva.com/g09/k_external.htm)
55. GauOpen: Interfacing to Gaussian 16 (v2), 访问时间为 四月 24, 2025， [https://gaussian.com/interfacing/](https://gaussian.com/interfacing/)
