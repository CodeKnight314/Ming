# Reference 8

## Prompt

能否给我提供一份详尽的报告，分析机器学习或者深度学习在优化材料元素组合配比以实现最佳的材料性能方面的研究进展和模型应用现状。请包括活跃的研究课题组，该课题组具体研究方向，已发表的相关论文，使用的数据库分析，模型准确度评估，面临的挑战以及对应的模型可行性分析。最后，请详细分析基于现有的研究现状，评估此领域距离实现理想模型的大规模应用和产业化还有多远

## Article

# **机器学习与深度学习在优化材料成分以实现目标性能方面的研究进展与应用现状分析报告**

## **1\. 引言**

### **1.1 范式转变：机器学习/深度学习通过成分优化加速材料发现**

传统上，新材料的发现过程往往依赖于研究人员的直觉、经验以及大量的试错实验 1。这种方法不仅耗时、成本高昂 3，而且在探索广阔的材料成分空间时效率低下 8。为了应对日益增长的技术需求，迫切需要一种能够加速新材料发现和探索的新方法 10。

近年来，机器学习（ML）和深度学习（DL）作为一种颠覆性的技术，为材料科学领域带来了新的研究范式 11。这些数据驱动的方法能够显著降低计算成本、缩短研发周期并提高预测精度 10，被认为是材料基因组工程（MGE）框架内的“第四范式” 11。其核心能力在于学习和理解材料的成分、结构、工艺和性能之间复杂且通常非线性的关系 14。通过这种能力，ML/DL模型可以有效地预测材料性能、优化材料成分组合，甚至进行逆向设计，即根据所需性能直接生成候选材料结构 18。

这一转变标志着材料研发从传统的高通量筛选（无论是基于密度泛函理论（DFT）等第一性原理计算 10 还是实验 20）向知识驱动的信息学方法 3 和直接生成候选材料 1 的演进。这种转变不仅仅是速度的提升，更重要的是，它代表了从依赖偶然发现或穷举搜索到目标导向设计的根本性变革。ML模型通过学习和封装复杂的物理化学规律 14，使得研究人员能够更直接地探索材料空间，根据预设的目标性能来指导材料的搜索或生成过程 18，从而根本上改变了材料发现的工作流程。

### **1.2 报告目标与范围**

本报告旨在基于现有研究资料，全面、深入地分析机器学习与深度学习在通过优化材料元素组合以实现特定材料性能方面的研究进展、模型应用现状、面临的挑战以及产业化前景。报告将重点探讨活跃的研究方向和课题组、常用的数据库和模型、模型评估方法、关键挑战及其应对策略，并评估该领域距离大规模工业应用的距离。

## **2\. ML/DL在材料成分优化中的应用现状**

### **2.1 加速研发周期：效率提升与成本降低**

ML/DL技术在材料研发中的一个核心优势是显著提高效率和降低成本。与传统的基于第一性原理（如DFT）的计算方法相比，ML模型可以在保证一定精度的前提下，大幅减少计算资源消耗，特别是对于需要高精度计算的复杂体系 10。这使得研究人员能够更快地评估大量候选材料。

通过快速预测材料性能，ML/DL技术能够有效缩短材料的研发周期 10。模型可以替代部分传统实验或计算模拟，或者与之协同工作，用于分析材料结构、预测材料性能，从而更高效、更准确地开发新型功能材料 8。这种加速效应有助于降低研发过程中的人力和物力成本 10。此外，ML模型能够指导实验设计，将研究重点放在最有潜力的候选材料上 20，例如，商业软件Alchemite™声称可以将实验时间和成本减少50-80% 24。

处理高维数据的能力也是ML/DL的一大优势 11。材料的性能往往受到众多因素（成分、结构、工艺参数等）的影响，形成一个高维复杂的空间。ML方法能够有效地处理这些高维数据，从中提取有意义的模式和关联，帮助研究人员在材料设计中做出更有效的决策 11。

从实践角度看，ML/DL在近期的主要价值体现为对现有研发流程的“效率放大”，而非完全取代。这些技术优化了资源（计算、实验、人力）的分配 10，通过充当强大的过滤器和预测器来指导实验方向 8，从而在现有研发框架内实现效率的显著提升。

### **2.2 关键应用领域**

ML/DL技术已广泛应用于众多材料体系的成分优化和性能预测，展现出强大的潜力。以下是一些关键的应用领域：

* **合金设计：** ML/DL被用于预测合金的各种性能，如强度、延展性 18、稳定性、表面成分 25 等，并指导新型合金的设计，例如高熵合金（HEAs） 10、增材制造（AM）合金 24、高温合金 24 和三元合金 27。研究人员利用ML模型探索成分-结构-性能关系，以优化合金配比，获得特定应用所需的性能 11。  
* **催化剂发现：** 这是ML/DL应用的一个非常活跃的领域。模型被用来加速发现和优化用于各种化学反应（如氨合成 20、费托合成、甲醇合成 30、析氢反应（HER） 31、水氧化 27 等）的均相和非均相催化剂 20。应用包括筛选大量候选催化剂 20、预测催化活性、选择性和稳定性 18、预测吸附能 31，甚至直接从实验动力学数据中阐明反应机理 30。  
* **聚合物材料：** ML/DL用于优化聚合物配方和加工工艺，以获得所需的性能，如导电性、机械强度、生物降解性 34 等 35。这包括电子聚合物 37、复合材料 15 和可持续聚合物 41 的设计。由于聚合物的加工-性能关系非常复杂 38，自主实验室（如Polybot 37）与ML的结合显示出巨大潜力。  
* **能源材料：** ML/DL在寻找和优化用于能源存储和转换的材料方面发挥着关键作用，包括超导材料 10、热电材料 10、光伏材料 10、电池材料（如电极材料、固态电解质） 17 和燃料电池材料 21。  
* **其他领域：** ML/DL的应用还扩展到陶瓷 24、功能磁性材料 7、混凝土 24、涂层 24、高能材料 8 以及光电材料 13 等。

总的来说，ML通过建立成分、相组成、结构、工艺与宏观性能（如力学、热学、电学、电化学等）之间的关联，实现了材料的优化设计和性能提升 11。

### **2.3 与材料基因组工程（MGE）的整合**

ML/DL技术被视为材料基因组工程（MGE）这一宏伟计划中的关键工具 11。MGE旨在通过整合计算、实验和数据科学方法，加速材料的发现、开发和部署。在高通量计算和实验产生海量数据的背景下 14，ML/DL成为了处理和解析这些大数据的核心引擎，代表了数据驱动的“第四范式” 11。

MGE的核心是建立“工艺-结构-性能”之间的关联，而ML模型正是通过学习这些关联来实现性能预测和材料设计 14。因此，ML/DL技术为实现MGE的目标提供了强大的分析能力。可以说，ML/DL是使MGE产生的大量数据变得可操作的关键分析引擎。没有ML/DL对海量高维数据的有效处理和模式挖掘能力 10，高通量方法产生的庞大数据可能变得难以管理和利用，从而阻碍MGE目标的实现。ML/DL提供了必要的工具，将MGE的数据生成能力转化为实际的设计指导和知识发现。

## **3\. 主要研究方向与前沿课题组**

随着ML/DL在材料科学领域的深入应用，多个研究方向涌现出活跃的研究团队和代表性项目。

### **3.1 重点领域：合金设计**

合金设计的目标是利用ML/DL预测成分、结构与性能（如强度、延展性 18、稳定性、表面成分 25）的关系，并设计具有特定性能的新型合金，如高熵合金 10、增材制造合金 24 和高温合金 24。无监督学习方法如聚类和降维被用于分析高维合金数据 29，而生成式人工智能则用于全新的合金设计 28。

**表1：机器学习用于合金设计的主要研究团队/项目**

| 团队/机构 | 具体研究方向/关键项目/技术 | 相关文献来源 |
| :---- | :---- | :---- |
| MIT 土木与环境工程系 (Buehler/LAMM) | AtomAgents：用于自主合金设计的多智能体AI平台，结合物理感知，可动态生成模拟数据 28 | 28 |
| 多伦多大学 (Zou Group) | 用于高熵/纳米晶合金发现、力学行为预测、增材制造工艺优化、基于计算机视觉的工艺监控的机器学习 26 | 26 |
| 澳大利亚国立大学/迪肯大学 (Barnard) | 应用无监督学习（聚类、降维、流形学习）分析高维合金数据，识别潜在规律，优化性能 29 | 29 |
| 卡内基梅隆大学 (CMU) | Kitchin（表面偏析建模 25）、Ulissi（金属间化合物、计算工具开发 25、Open Catalyst项目 45）、Gellman（实验验证平台、DMREF项目 25）、深度强化学习应用 25 | 25 |
| Intellegens (商业公司) | Alchemite™平台：源于与劳斯莱斯的合金项目，应用于增材制造、稀疏数据处理、实验设计（DOE）、研发洞察 24 | 24 |
| 微软研究院 (MatterGen/Sim) | MatterGen：用于新颖合金设计（如高体模量材料）的生成式AI；MatterSim：用于稳定性和性能预测的AI模拟器 21 | 21 |
| 奥钢联 (Voestalpine) (工业用户) | 应用Alchemite™进行材料和工艺创新，增材制造合金设计，工艺参数优化，质量保证 24 | 24 |
| 劳斯莱斯 (Rolls-Royce) (工业合作者) | 合作开发Alchemite™用于高温合金设计 24 | 24 |
| 美国国家航空航天局 (NASA) (工业用户) | 应用Alchemite™进行材料和部件设计 24 | 24 |
| 安赛乐米塔尔/OCAS (工业用户) | 应用Alchemite™进行钢性能预测 24 | 24 |

### **3.2 重点领域：催化剂发现**

该领域旨在利用ML/DL加速识别和优化用于各种化学过程的催化剂 20，预测催化剂的活性、选择性、稳定性 18，并理解反应机理 30。ML被用于筛选大量候选物 20，预测关键描述符（如吸附能 31），甚至直接从动力学数据推断机理 30。

**表2：机器学习用于催化剂发现的主要研究团队/项目**

| 团队/机构 | 具体研究方向/关键项目/技术 | 相关文献来源 |
| :---- | :---- | :---- |
| 曼彻ster大学 (Larrosa/Bures) | 基于ML/DL直接从动力学数据预测反应机理（包括活化/失活），开发实验设计工具，与BP合作（费托合成等） 30 | 30 |
| 堪萨斯大学 (CataLST NRT项目) | 结合AI与人类智能的催化剂发现培训项目；CataLST方法论：文献编目（Catalog）、ML学习（Learn）、计算搜索（Search）、实验测试（Test）；关注非均相催化，数据挖掘，NLP 47 | 47 |
| 卡内基梅隆大学 (CMU) | Ulissi（Open Catalyst项目 \[OC20, OC22数据集\]，用于大体系的GNN，ML势函数 27）、Kitchin（表面成分ML算法 25，描述符学习 31）、Gellman（实验验证 25） | 25 |
| 澳大利亚国立大学 (ANU) | 无监督学习用于合金设计（可能应用于催化剂） 29 | 29 |
| 微软研究院 (MatterGen/Sim) | MatterGen：用于新颖催化剂设计的生成式AI；MatterSim：用于稳定性和性能预测的AI模拟器 21 | 21 |

### **3.3 重点领域：聚合物配方与加工**

聚合物科学领域利用ML/DL优化配方和加工条件，以获得目标性能，如导电性、力学性能、生物降解性等 34。由于聚合物的加工-性能关系极其复杂 38，自主实验室（如Polybot）与ML的结合显示出巨大潜力 37。应用实例包括预测聚合物介电性能 13、增材制造聚合物的力学性能 36 以及设计可持续聚合物配方 41。

**表3：机器学习用于聚合物科学的主要研究团队/项目/公司**

| 团队/机构/公司 | 具体研究方向/关键项目/技术 | 相关文献来源 |
| :---- | :---- | :---- |
| 阿贡国家实验室/芝加哥大学 (Xu, Chan, Vriza) | Polybot：用于电子聚合物加工的AI驱动自主实验室，优化导电性和缺陷，多目标优化，共享开源数据 37 | 37 |
| Matmerize (商业公司) | PolymeRize平台：用于聚合物性能预测和可持续配方生成的深度学习平台（例如，生物基塑料替代品），加速开发（例如，CJ Biomaterials PHACT项目） 41 | 41 |
| 佐治亚大学 (UGA) | ML/AI优化配方/制造（增材制造、纳米复合材料），传感器反馈回路，模拟/数字孪生，实验设计（DOE）/主成分分析（PCA）对比试错法，纳米结构理解，流动化学反应优化 35 | 35 |
| 多个研究团队 (通用应用) | 使用ANN、SVR、RF等模型预测增材制造聚合物（ABS、PLA）的力学性能（拉伸强度、模量） 36 | 36 |
| 多个研究团队 (通用应用) | 预测聚合物的介电性能 13 | 13 |

在这些不同的研究方向中，一个共同的趋势是将ML模型开发与大规模模拟数据生成（如CMU的Open Catalyst项目 45）或自动化/高通量实验（如Polybot 37、CMU/Gellman的平台 25、堪萨斯大学的CataLST 47）紧密结合。这表明该领域认识到，仅靠ML模型本身不足以取得突破，必须辅以强大的数据生成策略。例如，CMU的研究明确将计算ML（Ulissi, Kitchin）与实验验证平台（Gellman）配对 25。Polybot的设计目标就是由AI驱动的自主数据生成 37。堪萨斯大学的CataLST模型包含了“学习”（ML）和“测试”（实验）两个阶段 47。这种模式反映了一种成熟的理解，即进展需要通过自动化等手段，有效地闭合“预测-验证”的循环。

## **4\. 基础支撑：材料数据库**

### **4.1 数据库在材料信息学中的作用**

材料数据库是机器学习模型不可或缺的“燃料” 12，它们提供结构化的信息 49，是训练预测模型的基础。与简单的文件集合不同 49，数据库能够有效管理海量数据，提供标准化的、灵活快速的数据访问模式，支持多用户并发访问，并实施必要的安全协议 49。

数据库在实现高通量（HT）第一性原理计算方面发挥了核心作用，它们存储了大量已知和假设材料的计算属性 2。这使得研究人员能够智能地查询数据库，寻找具有所需性能的材料，从而摆脱了传统研发中的猜测工作 2。此外，自然语言处理（NLP）技术也被用于从科学文献中提取数据，以构建和扩充材料数据库 7。

### **4.2 主要材料数据库概览**

材料信息学领域依赖于多个关键的开放获取数据库：

* **Materials Project (MP):** 专注于提供通过第一性原理计算得到的已知和预测的无机固体、分子、电池材料和催化剂的属性数据（结构、热力学、电子、力学、介电、磁性等）。提供网页访问、强大的分析工具（如相图绘制）和应用程序接口（API） 49。数据库包含超过17.8万种材料和57.7万个分子 50，并提供通过NLP从文献中提取的合成路线信息 50。  
* **AFLOW (Automatic FLOW for Materials Discovery):** 这是一个大型的DFT计算性质数据库，包含超过350万个条目和超过7.34亿个计算性质 55。涵盖无机晶体的热力学、电子、力学、热学、振动等多种性质。提供REST API（AFLOW-API）和AFLUX搜索语言进行数据访问 2，并包含一个包含1100多种结构原型的百科全书 55。  
* **OQMD (Open Quantum Materials Database):** 一个高通量DFT数据库，最初包含约30万次计算 59，目前约有70万种材料 60。重点关注材料的热力学稳定性（生成能、距离凸包能量），包含来自ICSD的结构和常见晶体结构的装饰相。其目标之一是预测可能存在的、先前未知的稳定化合物 59。提供REST API和网页界面访问 2。  
* **NOMAD (NOvel MAterials Discovery):** 一个更广泛的数据存档项目，旨在托管、共享和重用来自各种来源（模拟、实验）的计算材料科学数据。NOMAD-Ref作为一个参考数据集 51。该项目强调数据的FAIR（可发现、可访问、可互操作、可重用）原则 63。  
* **Materials Cloud:** 一个用于共享计算材料科学资源（数据、工具、工作流）的平台，由AiiDA驱动以追踪数据来源和计算历史。提供Discover（发现）、Explore（探索）和Archive（存档）等功能模块。同样强调FAIR数据原则和研究的可重复性 63。  
* **其他提及的数据库:** 包括Crystallography Open Database (COD) 49、Materials Data Facility (MDF) 49、Citrination 49、MatWeb 49、Computational Materials Repository (CMR) 2、Theoretical Crystallography Open Database (TCOD) 2 等。

### **4.3 数据类型与API访问机制**

这些数据库通常提供以下类型的数据：

* **成分/化学计量学:** 材料的化学组成 49。  
* **晶体结构:** 晶格参数、原子坐标、空间群、结构原型等 49。  
* **计算性质:** 生成能、距离凸包能量（稳定性）、能带隙、态密度、密度、弹性张量（体模量、剪切模量）、介电常数、磁性、热力学性质（焓、熵、热容）、振动性质（德拜温度）等 49。  
* **计算细节:** DFT计算参数（如泛函、截断能、k点）、收敛信息、计算耗时等 52。  
* **实验数据:** 通常通过链接到ICSD等外部数据库或通过用户贡献（如MPContribs）提供 53。  
* **合成信息:** 例如Materials Project的合成浏览器提供从文献中提取的合成路线 50。

访问这些数据库的主要方式是通过**REST API** 2。研究人员可以使用编程语言（如Python）和特定的客户端库（如MPRester 50、qmpy\_rester 60）或查询语言（如AFLUX 57）来批量检索和处理数据。通常需要注册并获取API密钥才能进行认证访问 53。

**表4：主要计算材料数据库比较 (Materials Project, AFLOW, OQMD)**

| 特征 | Materials Project (MP) | AFLOW | OQMD |
| :---- | :---- | :---- | :---- |
| **主要关注点** | 广泛的计算性质（热力学、电子、力学等），覆盖无机固体、分子、电池、催化剂。已知和预测材料。合成路线。50 | 超大规模DFT计算，广泛的性质覆盖（热力学、电子、力学、热学、振动）。关注无机晶体。结构原型百科全书。55 | 重点关注DFT计算的热力学稳定性（生成能、E\_hull）。基于ICSD结构和常见结构装饰相。预测新的稳定化合物。59 |
| **数据范围/规模** | 约17.9万种材料，约57.8万个分子（据50）。数据库持续增长。 | 超过350万个条目，超过7.34亿个性质（据55）。是提及的数据库中最大的之一。 | 最初约30万次计算59，目前约70万种材料60。 |
| **关键数据类型** | 结构、生成能、E\_hull、能带隙、弹性常数、介电常数、磁性、合成路线、分子数据、电池电极数据。50 | 结构、生成能、能带隙、态密度、弹性模量、热学性质（德拜温度、热容）、振动性质、Bader电荷、结构原型。55 | 结构、生成能、E\_hull（稳定性）、能带隙、体积、磁矩。主要侧重热力学性质。59 |
| **访问方式** | 网页门户、REST API（推荐使用mp-api Python客户端）、OPTIMADE API。50 | 网页门户（搜索、原型库）、REST API、AFLUX搜索API、AFLOW-ML界面。55 | 网页门户（搜索）、REST API（qmpy\_rester Python包装器）、兼容OPTIMADE API。60 |
| **独特功能** | 集成分析工具（相图、材料浏览器）、合成浏览器、MPContribs用户数据集成、分子浏览器。50 | 结构原型百科全书、AFLOW-CHULL（凸包构建）、AFLOW-ML模型、标准化的计算框架。55 | 强调评估DFT计算精度和预测新稳定相。开源qmpy基础设施。59 |

### **4.4 数据库整合 (OPTIMADE)**

材料数据领域的一个显著问题是数据资源的碎片化 2。为了解决这个问题，OPTIMADE（Open Databases Integration for Materials Design）联盟应运而生。该联盟致力于开发一个通用的API标准，以促进不同材料数据库之间的互操作性和可访问性 2。Materials Project和OQMD等主要数据库已经支持OPTIMADE接口 54。

OPTIMADE的出现反映了该领域的成熟，认识到孤立的数据仓库阻碍了进展。标准化对于构建更强大、能够利用多个数据源的ML模型和工作流至关重要。这种互操作性有助于缓解ML面临的一个关键挑战——数据稀缺性和多样性不足，从而有望提升模型的鲁棒性和泛化能力。

## **5\. 核心技术：ML/DL模型与评估**

### **5.1 数据表示与特征工程**

将材料信息（如成分、结构、图像、光谱等）转化为机器学习模型能够处理的数值形式（称为表示、特征、描述符或指纹）是应用ML/DL的首要且关键的步骤 8。

特征工程，即对影响材料性能的因素进行编码并选择最优特征子集的过程，对模型性能至关重要 18。不恰当的特征选择或冗余信息会严重影响模型的预测能力 8。常见的表示策略包括：

* **基于成分的特征：** 如化学计量比、元素的物理化学属性（离子半径、电负性、电离能、氧化态等） 13。  
* **基于结构的特征：** 如利用晶体图（节点代表原子，边代表键）表示原子连接性 66，或包含键长、键角等几何信息 69。  
* **学习得到的表示：** 现代ML，特别是深度学习模型，能够直接从原始数据中学习有效的表示，而无需手动设计特征 14。  
* **文本表示：** 如使用SMILES（简化分子线性输入规范）字符串表示分子 70，或直接使用自然语言描述材料 20。

特征工程常常会产生高维度的特征空间 18，这对于样本量有限的材料科学数据来说是一个挑战。因此，需要采用特征降维技术（如主成分分析PCA）或特征选择方法（如基于稀疏学习的SISSO算法 18）来筛选出最相关的特征 74。

特征工程是当前材料信息学的一个瓶颈，也是活跃的研究领域。近年来，研究趋势正朝着利用深度学习模型（如图神经网络GNNs或Transformer）自动学习表示的方向发展 14。这些模型可以直接处理晶体结构图或分子序列等原始数据，旨在克服手动设计描述符的局限性，学习到更有效、更具泛化能力的材料表示。

### **5.2 常用机器学习模型**

除了先进的深度学习模型外，一些传统的机器学习算法在材料科学中仍然得到广泛应用，尤其是在数据量较小或需要模型可解释性的场景下：

* **线性回归/LASSO：** 常用于构建基线模型或需要高可解释性的模型 36。LASSO通过正则化进行特征选择。  
* **支持向量机/回归 (SVM/SVR)：** 在处理小样本、高维度数据方面表现良好，常用于分类和回归任务 36。  
* **随机森林 (RF)：** 一种基于决策树的集成学习方法，鲁棒性较好，不易过拟合，能够提供特征重要性评估 76。  
* **K-近邻 (KNN)：** 一种简单的非参数方法，基于实例的学习 36。  
* **梯度提升机 (GBM) / XGBoost：** 强大的集成学习算法，通常能获得较高的预测精度 36。  
* **高斯过程 (GP)：** 一种基于贝叶斯理论的非参数模型，能够提供预测的不确定性估计 1。  
* **人工神经网络 (ANNs) / 多层感知器 (MLPs)：** 基础的神经网络模型，能够学习非线性关系 36。

这些模型已被应用于预测各种材料性能，如增材制造部件的力学强度 36、混凝土性能 44 和化学品毒性 77。研究表明，在某些任务中，ANNs的表现优于传统的实验设计（DOE）回归模型 36 或响应面法（RSM） 44。

### **5.3 先进深度学习模型**

随着数据量的增长和计算能力的提升，深度学习模型在材料科学领域展现出越来越强的能力。

* **图神经网络 (GNNs):**  
  * **适用性：** GNNs特别适合处理材料数据，因为它们能够自然地将原子（节点）和化学键（边）表示为图结构，从而有效捕捉原子的局部化学环境和相互作用信息 66。  
  * **代表性架构：** 包括晶体图卷积神经网络（CGCNN） 67、材料图网络（MEGNet） 69、SchNet 78、原子线图神经网络（ALIGNN） 69、图注意力网络（GATGNN） 80、DimeNet/GemNet 69 等。  
  * **应用：** GNNs已被广泛用于预测材料的结构和电子性质，如密度、生成能、稳定性、能带隙等，并在许多任务上取得了较高的预测精度（例如，某些性质的 R2 值超过0.96 66，尽管稳定性和带隙的预测精度可能较低） 13。它们也是发现新型无机材料的有力工具 13。  
  * **研究前沿：** 当前研究方向包括构建更深层的GNN以处理复杂性（解决过平滑问题） 80、开发预训练策略（如学习力场）以提升泛化能力 81、利用集成学习提高模型鲁棒性 69、应用迁移学习处理数据稀疏问题 31、增强模型可解释性 67 以及通过反演GNN直接生成目标结构 82。  
  * **挑战：** GNNs在泛化到大型、多样化或分布外（out-of-distribution）的材料体系时仍面临挑战 31，模型的可扩展性也是一个问题 80。同时，也有观点对其普适性持保留态度 83。  
* **生成模型:**  
  * **目标：** 实现材料的逆向设计，即根据所需的目标性能直接生成全新的材料结构或成分，而不是仅仅预测给定结构的性能 1。这类模型旨在探索整个未知的材料空间 21。  
  * **常见类型：**  
    * **进化算法 (EAs)：** 如遗传算法（GAs）和粒子群优化（PSO）。常用于晶体结构预测（例如，CALYPSO软件包 1）、优化材料性能（如硬度、纳米颗粒偏析） 1。可以通过ML代理模型加速（如MLaGA） 1。  
    * **生成对抗网络 (GANs)：** 由生成器和判别器组成，通过对抗训练生成逼真的样本 22。  
    * **变分自编码器 (VAEs)：** 将数据编码到低维潜空间，再从潜空间解码生成新样本 84。  
    * **扩散模型：** 通过逐步添加噪声破坏数据，然后学习逆向去噪过程来生成样本。能够生成高质量、多样化的样本，且无需复杂的对抗训练 22。微软的MatterGen即采用了扩散模型 21。  
  * **挑战：** 确保生成的材料具有热力学稳定性和可合成性是主要难点 22。如何实现受控生成（满足特定约束或多目标优化） 22、如何生成真正新颖且超越训练数据分布的高性能材料（外插能力） 22 以及如何快速有效地验证和筛选大量生成候选物 22 都是亟待解决的问题。  
* **Transformer模型:**  
  * **核心机制：** 源自自然语言处理（NLP），其核心是自注意力（self-attention）机制，能够捕捉序列数据中的长程依赖关系和上下文信息 70。  
  * **材料科学应用：** 通过将材料表示为序列数据，如SMILES字符串 70、SELFIES字符串 88、标记化的晶体学信息（如MatInFormer 88）或自然语言描述 72，Transformer模型被应用于材料领域。  
  * **应用实例：** 分子性质预测 70、化学反应预测/逆合成 70、聚合物性质预测（如TransPolymer 72）、合金性质预测（如AllyBERT 72）。  
  * **代表性架构：** 基于NLP领域的模型进行调整，如BERT、RoBERTa、ALBERT、ELECTRA 87、BART 70、GPT 86、T5 70 等。  
  * **优势：** 潜力在于实现高预测精度、提供一定的可解释性（通过注意力可视化 89）、能够利用大规模预训练模型进行迁移学习 71、以及有效处理序列化数据。  
  * **挑战：** 模型性能高度依赖于训练数据的质量和多样性 70。如何选择最优的材料序列表示方法（如SMILES存在局限性 70）、设计有效的预训练任务和策略、处理模型的可扩展性问题以及建立统一的基准测试标准 71 都是当前面临的挑战。如何将Transformer与GNN或物理知识更有效地结合也是一个研究方向 90。

### **5.4 模型性能评估指标**

评估机器学习模型的性能对于理解其预测能力和泛化能力至关重要 91。对于材料属性预测这类回归任务，常用的评估指标包括：

* 平均绝对误差 (Mean Absolute Error, MAE): 计算预测值与真实值之间绝对误差的平均值。MAE的单位与目标变量相同，易于理解，并且对异常值不敏感，因为它对所有误差的惩罚是线性的 91。计算公式为：  
  MAE=n1​∑i=1n​∣yi​−y^​i​∣  
  其中 yi​ 是真实值，y^​i​ 是预测值，n是样本数量。  
* 均方误差 (Mean Squared Error, MSE): 计算预测值与真实值之间误差平方的平均值。由于进行了平方操作，MSE对较大的误差给予更高的权重，因此对异常值更敏感。其单位是目标变量单位的平方。MSE是可微的，常用于模型优化过程 91。计算公式为：  
  MSE=n1​∑i=1n​(yi​−y^​i​)2  
* 均方根误差 (Root Mean Squared Error, RMSE): 是MSE的平方根。RMSE的单位与目标变量相同，比MSE更易于解释。与MSE类似，RMSE也对较大的误差给予更高的权重，对异常值敏感 91。计算公式为：  
  RMSE=n1​∑i=1n​(yi​−y^​i​)2​  
* 决定系数 (R-squared, R2): 表示模型解释的目标变量方差的比例，衡量模型的拟合优度。R2 的值通常在0到1之间（对于普通最小二乘法拟合），值越接近1表示模型拟合得越好。R2 是一个无量纲的相对指标，便于比较不同模型在同一数据集上的表现，或比较模型在不同尺度数据集上的表现 91。但需要注意，R2 值高并不总意味着模型是好的，例如过拟合的模型也可能有高 R2 91。R2 被认为是一个信息量丰富且真实的指标 97。计算公式为：  
  R2=1−∑i=1n​(yi​−yˉ​)2∑i=1n​(yi​−y^​i​)2​  
  其中 yˉ​ 是真实值的平均值。  
* **其他指标：** 还包括平均绝对百分比误差（MAPE）、对称平均绝对百分比误差（SMAPE） 94 和均方根对数误差（RMSLE） 91 等。

选择哪个指标取决于具体的应用场景和对误差的容忍度。如果需要重点关注大误差，MSE/RMSE是合适的选择；如果希望所有误差被同等对待或减轻异常值的影响，MAE更优 91。通常建议同时考察多个指标以全面评估模型性能 92。

除了选择合适的指标，还必须采用恰当的验证策略，如**交叉验证**（例如5折交叉验证 76），以评估模型的稳定性和泛化能力，避免模型在训练集上过拟合而在未见数据上表现不佳 18。

值得注意的是，虽然 R2 提供了一个标准化的拟合优度度量，但MAE和RMSE提供了关于预测误差绝对大小（以目标属性的单位表示）的信息。在实际应用中，理解模型的拟合程度（R2）和典型预测误差的大小（MAE/RMSE）都至关重要。例如，一个预测材料强度的模型可能 R2=0.9，看似很好，但如果其RMSE为50 MPa，对于需要±10 MPa公差的应用来说可能毫无价值。因此，仅凭 R2 不足以判断模型的实用性 91，必须结合考虑MAE/RMSE等尺度相关的误差指标 91。

## **6\. 导航障碍：挑战与模型可行性**

尽管ML/DL在材料成分优化方面取得了显著进展，但仍面临诸多挑战，这些挑战影响着当前模型的可行性和未来的发展方向。

### **6.1 数据限制**

数据是驱动ML模型的基石，但在材料科学领域，获取高质量、大规模的数据本身就是一个重大挑战。

* **数据稀缺性（“小数据”问题）：** 这是最核心的挑战之一。ML模型，尤其是深度学习模型，通常需要大量数据进行训练才能达到良好的性能 12。然而，材料数据的获取（无论是通过实验还是高精度的计算模拟）成本高昂、耗时费力 3。因此，许多材料科学数据集的规模相对较小（例如，约57%的数据集样本量小于500 74），导致特征维度与样本量 74 或模型参数与样本量 74 之间比例失衡，极易引发模型过拟合 77。这与许多拥有“大数据”的其他AI应用领域形成鲜明对比 101。  
* **数据质量与一致性：** 由于缺乏统一的实验或计算标准 4，来自不同来源的数据质量参差不齐，存在不一致性 3。数据中可能包含噪声 103，影响模型训练效果。  
* **数据偏差：** 现有数据库普遍存在偏差。例如，它们倾向于收录已成功合成的、稳定的材料，而缺乏“负样本”数据（如不稳定、无法合成的材料或失败的实验记录） 19。这种偏差使得训练能够准确预测材料不稳定性或不可合成性的模型变得困难 19。科学出版物倾向于报道成功结果也加剧了这个问题 101。  
* **数据多样性与可获得性：** 数据集可能缺乏足够的多样性，无法覆盖广阔的材料空间 84。同时，许多有价值的工业数据由于保密原因难以获取 4。数据来源多样、格式各异，也给数据整合和标准化带来了挑战 101。  
* **应对策略：**  
  * **数据增强：** 通过算法生成虚拟样本来扩充数据集（如SMOTE、GMM 77），或使用非标准化的表示（如非规范SMILES 70）。  
  * **迁移学习：** 将在数据丰富的源任务上学到的知识迁移到数据稀疏的目标任务上 14。例如，提出了混合专家（mixture of experts）框架 102。  
  * **半监督学习 (SSL)：** 结合少量标记数据和大量未标记数据进行训练。例如，正样本与未标记样本学习（PU Learning）已用于预测材料可合成性 19，师生模型（Teacher-Student models）也被探索用于解决负样本缺失问题 19。这些方法依赖于一些基本假设，如平滑性、低密度分离和流形假设 19。  
  * **主动学习/贝叶斯优化/自主实验室：** 通过智能算法选择信息量最大的实验来执行，从而高效地生成高质量数据，实现“闭环”优化 3。  
  * **元学习（“学习如何学习”）：** 训练能够快速适应新任务、仅需少量数据即可学习的模型 77。  
  * **多任务学习：** 同时在多个相关任务上训练模型，共享知识 77。  
  * **特征降维与领域知识融合：** 减少特征数量以匹配小数据集规模，并将物理、化学等领域知识融入模型构建或特征选择过程，以提高模型性能和数据效率 74。

### **6.2 模型自身的挑战**

除了数据问题，ML/DL模型本身也存在一些固有的挑战：

* **可解释性（“黑箱”问题）：** 许多高性能模型，特别是深度神经网络和复杂的集成模型，其内部决策过程不透明，难以理解模型做出特定预测的原因 8。这不仅阻碍了用户（尤其是领域专家）对模型的信任和采纳，也使得模型调试和从中提取科学见解变得困难 8。因此，发展可解释人工智能（XAI）方法（如SHAP、LIME、PDP 77）或构建本质上可解释的模型（如简化的线性模型 76、利用Transformer的注意力可视化 89）变得越来越重要。  
* **泛化能力：** 模型可能在训练数据或特定基准测试中表现优异，但在面对全新的、分布之外的材料或条件时性能急剧下降 22。过拟合风险始终存在，尤其是在数据稀疏的情况下 77。提升模型的鲁棒性 74 和外推能力是关键。  
* **计算成本：** 虽然ML通常比高精度DFT计算成本更低 10，但训练大型深度学习模型（尤其是生成模型或大型Transformer模型）仍然需要大量的计算资源 3。模型的推理（预测）可能很快，但训练阶段的成本不容忽视。量子计算被视为未来解决计算瓶颈的潜在途径 4。  
* **不确定性量化 (UQ)：** 在材料研发中，了解模型预测的不确定性或置信度对于做出可靠决策（例如，决定下一步进行哪个成本高昂的实验）至关重要 3。贝叶斯学习方法是解决UQ问题的一个重要方向 3。

### **6.3 特定领域的障碍**

在将ML/DL应用于具体的材料设计问题时，还存在一些领域特有的难题：

* **可合成性预测：** 预测一个计算设计的材料是否能在实验室中被实际合成出来，是当前面临的主要挑战之一，尤其对于生成模型而言 19。仅仅预测热力学稳定性（如生成能、距离凸包能量 Ehull​ 19）是不够的 31，动力学因素和实际的合成工艺约束同样重要 79。缺乏可靠的“不可合成”负样本数据是训练此类模型的关键障碍 19。目前正在开发基于PU学习、SSL等ML方法来应对这一挑战 19。逆合成路线预测也是一个相关的研究方向 5。  
* **稳定性预测：** 即便是预测材料的热力学稳定性，对于ML模型来说也并非易事 19。生成模型经常产生不稳定的结构 84。需要开发快速准确的验证模型来筛选生成结果 22。此外，为了更真实地预测材料在实际应用条件下的稳定性，还需要考虑温度、压力等外部因素的影响 23。  
* **多目标优化 (MOO)：** 实际应用中，材料通常需要同时满足多个性能指标，例如合金的强度和延展性 18，或催化剂的活性、选择性和稳定性 18。这些性能指标之间可能存在冲突和权衡关系，需要采用帕累托（Pareto）优化等方法寻找最优解集 18。ML可以帮助快速探索帕累托前沿，或者将多目标问题分解为多个单目标问题进行处理 18。这需要仔细设计模型架构（例如，采用多输出模型还是多个单输出模型） 18。阿贡国家实验室的Polybot系统在聚合物薄膜优化中成功应用了MOO策略 37。

### **6.4 与实验和领域知识的整合**

弥合计算预测与实验验证之间的鸿沟是推动该领域发展的关键 22。ML模型的预测结果最终需要通过实验来证实 25。

将物理、化学等领域知识融入ML模型，可以有效提高模型的性能、可解释性和数据效率 3。领域知识可以指导特征选择 74，或者作为约束条件加入模型训练中，使其预测更符合物理规律。MIT的AtomAgents项目就旨在构建物理感知的AI系统 28。

此外，“人在环路”（human-in-the-loop）的方法也值得探索，它可以将人类专家的经验和判断与ML的高效计算能力相结合 9。

### **6.5 当前模型可行性评估**

基于上述分析，可以对当前ML/DL模型在不同任务上的可行性进行评估：

* **性能预测：** 对于在训练数据覆盖范围内的材料性能预测任务，ML模型已展现出很高的可行性，其预测精度在某些情况下可以媲美甚至超越DFT计算 12。对于插值性质的任务，可行性很高。  
* **高通量筛选：** ML模型非常适合用于快速筛选由其他方法（如DFT计算、组合库）产生的大量候选材料，能够显著缩小实验或高成本模拟的搜索空间 8。可行性极高，是当前应用最广泛的场景之一。  
* **逆向设计（生成模型）：** 目前仍处于发展初期，面临着生成结构的稳定性、可合成性以及能否产生真正超越现有材料性能的新颖结构等方面的重大挑战 22。虽然前景广阔 21，但需要进一步的研发和验证。微软的MatterGen展示了潜力，但仍需更广泛的验证 21。  
* **机理推断：** 这是一个新兴的应用方向，已在均相催化动力学分析中显示出效果 32，并正在向非均相体系扩展 30。其可行性高度依赖于高质量动力学数据的可获得性。  
* **总体评估：** 当前，ML/DL技术在通过**性能预测**和**高通量筛选**来**加速**现有材料研发流程方面是高度可行的，并已产生实际价值。然而，通过生成模型实现真正意义上的**从头自主发现**革命性新材料的目标，在很大程度上仍处于探索阶段，面临诸多可行性障碍。

最大的可行性差距存在于**预测**（在已知数据/结构范围内进行插值）和**真正的发现**（外推到全新的、可合成的、高性能材料）之间。当前的ML技术擅长前者，但在后者上仍显不足。虽然基准数据集上的高预测精度 66 证明了其插值能力，但泛化能力差 31、生成结构的可合成性 19 和稳定性 84 问题，以及难以进行创造性设计（外推） 22 等持续存在的挑战，都表明利用现有ML技术可靠地探索和验证材料“未知领域”存在根本困难。生成模型旨在弥合这一差距 21，但其按需提供新颖实用材料的能力仍在发展和验证中。

## **7\. 弥合差距：工业应用与商业化**

### **7.1 当前工业应用格局**

机器学习正在被越来越多的工业界参与者所采纳，从初创公司到大型跨国企业都在探索其在材料研发中的应用 103。高管层对利用AI提升研发效率的兴趣日益浓厚 103。

一些具体的公司及其应用实例包括：

* **Intellegens:** 其Alchemite™平台被劳斯莱斯、奥钢联、NASA、安赛乐米塔尔/OCAS、AMRC、LLNL、Welding Alloys等多家公司用于合金、增材制造、塑料、复合材料、陶瓷、配方产品等领域 24。  
* **微软:** 开发MatterGen（生成式设计）和MatterSim（模拟与验证）工具，旨在推动电池、磁体、燃料电池等领域的材料创新，并积极与工业界合作 21。  
* **Matmerize:** 其PolymeRize平台利用深度学习进行聚合物性能预测和可持续配方设计，已应用于CJ Biomaterials 41。  
* **其他初创公司:** 如Machina Labs（航空航天合金的AI辅助成型） 41、Basetwo AI（增材制造工艺优化） 41、restor3d/NuVasive（AI设计医疗植入物） 41、Ayar Labs/Lightmatter（AI在光子学材料与器件中的应用） 41、Matgenix（提供ML/模拟服务的初创公司） 65。  
* **大型企业内部应用/平台:** 如Voestalpine 24、Henkel（孵化出Albert Invent平台） 103、Citrine Informatics、Fehrmann MaterialsX（商业平台，探索LLM应用） 103。

这些应用遍及航空航天、汽车、化工、能源、医疗健康、电子等多个行业 4。阿贡国家实验室等研究机构也在积极与工业界合作，利用其AI、超算（Exascale）、材料工程研究设施（MERF）和先进光子源（APS）等资源，加速能源材料的发现和产业化 42。

### **7.2 自主实验室/自驱动实验室 (SDLs) 的作用**

自主实验室（Self-Driving Labs, SDLs）是将自动化机器人平台与AI/ML技术相结合，实现实验设计、执行、分析的闭环自动化系统 9。

SDLs的核心目标是通过实现高通量、全天候的自主探索和优化，极大地加速材料发现过程 9。它们旨在让科学家能够“更聪明地失败，更快地学习，并花费更少的资源” 9。实例包括阿贡/芝加哥大学的Polybot（用于电子聚合物） 37、SLAC/斯坦福的CAMEO（用于外延纳米复合相变存储材料） 9 以及利物浦大学的移动机器人（用于催化剂探索） 107。

SDLs带来的益处显而易见：显著提高实验速度和通量 105、改善实验的可重复性 105、减少人力需求和相关成本 107、提高实验安全性 105，并且能够生成大量高质量、结构化的数据，这对于训练和改进ML模型至关重要 105。

因此，SDLs被视为弥合计算预测与实际验证之间差距的关键技术，是加速ML/DL技术在材料领域工业化应用的重要推动力 9。美国能源部（DOE）等机构已将其视为未来材料研究的重要方向 43。

### **7.3 商业化挑战**

尽管前景广阔，但将ML/DL技术和SDLs大规模商业化仍面临诸多挑战：

* **数据问题：** 数据质量、可用性、获取成本等问题依然突出 4。  
* **计算资源：** 高性能计算（HPC）的复杂性和成本 4。  
* **人才短缺：** 缺乏同时具备材料科学和数据科学专业知识的跨学科人才 4。  
* **知识产权与数据安全：** 在竞争激烈的市场中，如何保护专有数据和知识产权，同时促进必要的开放合作以加速创新，是一个难题 4。  
* **投资成本：** 建立SDLs、开发软件平台和维护HPC基础设施需要大量资金投入，这可能限制了小型研究机构和初创公司的参与 4。  
* **集成与工作流程：** 将新的ML工具和SDLs整合到现有的、复杂的工业研发流程和组织文化中并非易事 103。  
* **模型可靠性：** 确保模型在实际工业应用中的鲁棒性、泛化能力和预测精度至关重要 104。  
* **验证与放大：** 将实验室规模的发现成功转化为可行的工业化生产是一个巨大的挑战 42。  
* **领域特定难题：** 如前所述，准确预测可合成性等问题仍是重大障碍 22。

### **7.4 研究与产业化差距分析**

目前，学术界在ML/DL应用于材料科学方面取得了显著进展，工业界的兴趣和初步应用也在不断增长 65。然而，在学术演示（通常基于标准化的基准数据集）与广泛、可靠地将这些技术用于工业级的复杂材料**从头设计**之间，仍然存在明显的差距 23。

当前的工业应用更多地集中在利用ML来**优化**现有流程、**筛选**已知材料空间或**加速**特定的研发环节（如实验设计、工艺参数优化），而不是完全依赖AI进行革命性新材料的自主发现 24。

造成这种差距的主要原因正是7.3节中列出的挑战：数据瓶颈、高昂的成本、人才匮乏、知识产权顾虑、验证困难、集成障碍以及模型可靠性问题。SDLs虽然是重要的推动力，但其本身仍处于相对早期的发展阶段，需要大量的投资和专业知识才能有效部署和运行 43。实现SDL工作流程和数据的标准化也是未来的重要工作 105。

这种差距不仅仅是技术层面的，也涉及基础设施和文化层面。工业界需要的是经过充分验证、可扩展、具有成本效益、并且能够无缝集成到现有流程中的解决方案，以及能够管理这些先进技术的人才。此外，在将AI预测应用于高风险决策之前，建立对AI模型的信任也至关重要。SDLs的建设本身就是一项重大的基础设施投资 43，而模型的可解释性 8 和严格的验证 25 则是建立信任的基础。

## **8\. 未来展望与结论**

### **8.1 进展与挑战并存**

回顾过去几年的发展，ML/DL在材料科学领域取得了令人瞩目的成就。这些技术已成功应用于预测多种材料体系的性能 10，有效加速了材料筛选和优化过程 10。强大的新模型架构，如图神经网络、Transformer和生成模型不断涌现 21。同时，支撑这些研究的基础设施，包括开放数据库、API接口和自主实验室也在逐步建立和完善 2。

然而，挑战依然严峻。数据限制（稀缺性、质量、偏差、负样本缺失） 3、模型局限（可解释性差、泛化能力弱、外推困难） 8、可靠预测可合成性与稳定性的难题 22、多目标优化 18、与实验及领域知识的有效融合 25 以及高昂的实施成本和复杂性 4 仍然是制约该领域发展的关键因素。

### **8.2 领域成熟度与产业化路径评估**

目前，该领域正处于快速发展阶段，但距离实现广泛的、完全自主的工业级材料**发现**尚未完全成熟。其在**加速**现有研发流程（如筛选、性能预测、工艺优化）方面的成熟度相对较高。

通往大规模产业化的路径需要克服以下关键障碍：

* **完善数据基础设施：** 遵循FAIR原则 63，推动数据标准化 2，并设法获取更多高质量的负样本/失败实验数据 19。  
* **开发更优模型：** 需要更鲁棒、泛化能力更强、可解释性更好、并能有效融合物理/化学知识的模型 74。  
* **攻克可合成性预测难题：** 这是实现生成模型价值的关键 19。  
* **推广与降低SDL成本：** 提高自主实验室的可扩展性，降低其部署和运行成本 43。  
* **加强产学研合作：** 促进学术界、国家实验室和工业界之间的紧密合作与知识共享 23。  
* **培养跨学科人才：** 建立能够满足未来需求的材料信息学人才梯队 4。

### **8.3 核心遗留障碍**

总结而言，阻碍ML/DL在材料领域实现理想模型大规模应用和产业化的核心障碍主要包括：

1. **可合成性与稳定性瓶颈：** 准确预测计算设计的材料是否能够以及如何稳定存在并被成功合成，仍然是生成式方法面临的最大挑战 19。  
2. **泛化与外推能力：** 让模型能够超越训练数据的范围，可靠地预测全新、结构迥异的材料的性能 22。  
3. **数据生态系统建设：** 构建全面、高质量、标准化、易于访问的数据生态系统，特别是包含关键的失败案例和负样本数据 4。  
4. **集成与工作流程整合：** 将ML工具和SDLs无缝地融入复杂且已有的工业研发和生产流程中 103。  
5. **验证与信任建立：** 建立严格的验证标准和流程，增强工业界对AI驱动预测结果在关键应用中的信心 8。

### **8.4 预估时间框架与未来方向**

虽然难以给出精确的时间表，但可以预见，利用ML/DL**优化和加速**现有材料研发流程的应用将在未来5年内持续快速增长并普及。

然而，实现通过ML/SDLs进行可靠的、自主的**从头发现**革命性新材料，并将其广泛应用于工业界，可能是一个更长期的目标（或许需要10年以上），这取决于能否有效克服上述核心障碍。尽管如此，材料信息学有望将材料发现的时间从几十年缩短到几年甚至几个月 4。

未来的研究方向将可能集中在：开发融合物理/领域知识的混合模型 28；构建具有更强约束和更高生成质量的生成模型 22；发展功能更强大、集成度更高的自主实验室 105；利用大型语言模型（LLMs）进行数据提取、知识整合和人机交互 73；建立标准化的基准测试和评估体系 71；以及探索量子计算在材料模拟和设计中的潜力 90。

### **8.5 结论：变革潜力巨大**

机器学习和深度学习，特别是与自动化实验相结合时，拥有彻底改变材料科学研究和应用的巨大潜力 11。尽管挑战依然存在，但该领域的快速进展预示着一个未来：材料创新的周期将被显著压缩，从而能够更快地开发出满足能源、健康、电子和可持续发展等关键领域需求的先进材料。通过持续的研究投入、跨学科合作以及对核心挑战的攻坚，这一变革性的未来正逐步成为现实。

#### **引用的著作**

1. AI-driven materials design: a mini-review \- arXiv, 访问时间为 四月 24, 2025， [https://arxiv.org/html/2502.02905v1](https://arxiv.org/html/2502.02905v1)
2. Open Databases Integration for Materials Design \- CECAM, 访问时间为 四月 24, 2025， [https://www.cecam.org/workshop-details/open-databases-integration-for-materials-design-52](https://www.cecam.org/workshop-details/open-databases-integration-for-materials-design-52)
3. Knowledge-driven learning, optimization, and experimental design under uncertainty for materials discovery \- PubMed Central, 访问时间为 四月 24, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC10682757/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10682757/)
4. Future of Material Informatics in North America \- MarketsandMarkets, 访问时间为 四月 24, 2025， [https://www.marketsandmarkets.com/blog/SE/future-material-informatics-north-america](https://www.marketsandmarkets.com/blog/SE/future-material-informatics-north-america)
5. Retro-Rank-In: A Ranking-Based Approach for Inorganic Materials Synthesis Planning, 访问时间为 四月 24, 2025， [https://arxiv.org/html/2502.04289v1](https://arxiv.org/html/2502.04289v1)
6. AI-driven inverse design of materials: Past, present and future \- arXiv, 访问时间为 四月 24, 2025， [https://arxiv.org/html/2411.09429v1](https://arxiv.org/html/2411.09429v1)
7. Material informatics for functional magnetic material discovery \- AIP Publishing, 访问时间为 四月 24, 2025， [https://pubs.aip.org/aip/adv/article/14/1/015313/2993736/Material-informatics-for-functional-magnetic](https://pubs.aip.org/aip/adv/article/14/1/015313/2993736/Material-informatics-for-functional-magnetic)
8. of challenges for the machine learning in materials science. \- ResearchGate, 访问时间为 四月 24, 2025， [https://www.researchgate.net/figure/of-challenges-for-the-machine-learning-in-materials-science\_fig2\_355084977](https://www.researchgate.net/figure/of-challenges-for-the-machine-learning-in-materials-science_fig2_355084977)
9. On-the-fly closed-loop materials discovery via Bayesian active learning \- PMC, 访问时间为 四月 24, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC7686338/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7686338/)
10. Application of Machine Learning in Material Synthesis and Property Prediction, 访问时间为 四月 24, 2025， [https://www.researchgate.net/publication/373568100\_Application\_of\_Machine\_Learning\_in\_Material\_Synthesis\_and\_Property\_Prediction](https://www.researchgate.net/publication/373568100_Application_of_Machine_Learning_in_Material_Synthesis_and_Property_Prediction)
11. Applications of machine learning method in high-performance materials design: a review, 访问时间为 四月 24, 2025， [https://www.oaepublish.com/articles/jmi.2024.15](https://www.oaepublish.com/articles/jmi.2024.15)
12. Application of Machine Learning in Material Synthesis and Property Prediction \- PMC, 访问时间为 四月 24, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC10488794/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10488794/)
13. Machine Learning Approaches for Accelerated Materials Discovery \- AZoM, 访问时间为 四月 24, 2025， [https://www.azom.com/article.aspx?ArticleID=23290](https://www.azom.com/article.aspx?ArticleID=23290)
14. Representations of Materials for Machine Learning | Annual Reviews, 访问时间为 四月 24, 2025， [https://www.annualreviews.org/content/journals/10.1146/annurev-matsci-080921-085947](https://www.annualreviews.org/content/journals/10.1146/annurev-matsci-080921-085947)
15. Artificial Intelligence in Predicting Mechanical Properties of Composite Materials \- MDPI, 访问时间为 四月 24, 2025， [https://www.mdpi.com/2504-477X/7/9/364](https://www.mdpi.com/2504-477X/7/9/364)
16. Machine Learning for Structural Materials \- Annual Reviews, 访问时间为 四月 24, 2025， [https://www.annualreviews.org/doi/10.1146/annurev-matsci-110519-094700](https://www.annualreviews.org/doi/10.1146/annurev-matsci-110519-094700)
17. A Review of Performance Prediction Based on Machine Learning in Materials Science, 访问时间为 四月 24, 2025， [https://www.mdpi.com/2079-4991/12/17/2957](https://www.mdpi.com/2079-4991/12/17/2957)
18. Multi-objective optimization in machine learning assisted materials design and discovery, 访问时间为 四月 24, 2025， [https://www.oaepublish.com/articles/jmi.2024.108](https://www.oaepublish.com/articles/jmi.2024.108)
19. Materials synthesizability and stability prediction using a semi-supervised teacher-student dual neural network \- RSC Publishing Home, 访问时间为 四月 24, 2025， [https://pubs.rsc.org/en/content/articlehtml/2023/dd/d2dd00098a](https://pubs.rsc.org/en/content/articlehtml/2023/dd/d2dd00098a)
20. AI-Empowered Catalyst Discovery: A Survey from Classical Machine Learning Approaches to Large Language Models \- arXiv, 访问时间为 四月 24, 2025， [https://arxiv.org/html/2502.13626v1](https://arxiv.org/html/2502.13626v1)
21. MatterGen: A new paradigm of materials design with generative AI ..., 访问时间为 四月 24, 2025， [https://www.microsoft.com/en-us/research/blog/mattergen-a-new-paradigm-of-materials-design-with-generative-ai/](https://www.microsoft.com/en-us/research/blog/mattergen-a-new-paradigm-of-materials-design-with-generative-ai/)
22. Generative AI for Materials Discovery: Design Without Understanding, 访问时间为 四月 24, 2025， [https://www.engineering.org.cn/engi/EN/10.1016/j.eng.2024.07.008](https://www.engineering.org.cn/engi/EN/10.1016/j.eng.2024.07.008)
23. AI meets materials discovery: The vision behind MatterGen and MatterSim \- Microsoft, 访问时间为 四月 24, 2025， [https://www.microsoft.com/en-us/research/story/ai-meets-materials-discovery/](https://www.microsoft.com/en-us/research/story/ai-meets-materials-discovery/)
24. Machine learning for Materials R\&D \- intellegens, 访问时间为 四月 24, 2025， [https://intellegens.com/customers/materials/](https://intellegens.com/customers/materials/)
25. CMU becomes go-to place for machine learning in catalysis research, 访问时间为 四月 24, 2025， [https://www.cheme.engineering.cmu.edu/news/2019/12/03-ml-catalysis.html](https://www.cheme.engineering.cmu.edu/news/2019/12/03-ml-catalysis.html)
26. Zou Research Group \- Laboratory for Extreme Mechanics & Additive Manufacturing, 访问时间为 四月 24, 2025， [https://www.zou-mse-utoronto-ca.net/research](https://www.zou-mse-utoronto-ca.net/research)
27. John R. Kitchin (0000-0003-2625-9232) \- ORCID, 访问时间为 四月 24, 2025， [https://orcid.org/0000-0003-2625-9232](https://orcid.org/0000-0003-2625-9232)
28. Automating alloy design with advanced AI that can produce its own ..., 访问时间为 四月 24, 2025， [https://cee.mit.edu/automating-alloy-design-with-advanced-ai-that-can-produce-its-own-data-on-the-fly/](https://cee.mit.edu/automating-alloy-design-with-advanced-ai-that-can-produce-its-own-data-on-the-fly/)
29. Unsupervised learning and pattern recognition in alloy design \- RSC Publishing Home, 访问时间为 四月 24, 2025， [https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00282b](https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00282b)
30. Deep Learning-based mechanistic elucidation of processes mediated by heterogeneous catalysts at The University of Manchester on FindAPhD.com, 访问时间为 四月 24, 2025， [https://www.findaphd.com/phds/project/deep-learning-based-mechanistic-elucidation-of-processes-mediated-by-heterogeneous-catalysts/?p181356](https://www.findaphd.com/phds/project/deep-learning-based-mechanistic-elucidation-of-processes-mediated-by-heterogeneous-catalysts/?p181356)
31. Publications \- Ulissi Group, 访问时间为 四月 24, 2025， [https://ulissigroup.cheme.cmu.edu/publications/](https://ulissigroup.cheme.cmu.edu/publications/)
32. Organic reaction mechanism classification using machine learning \- PubMed, 访问时间为 四月 24, 2025， [https://pubmed.ncbi.nlm.nih.gov/36697863/](https://pubmed.ncbi.nlm.nih.gov/36697863/)
33. Organic reaction mechanism classification using machine learning | Request PDF, 访问时间为 四月 24, 2025， [https://www.researchgate.net/publication/367418407\_Organic\_reaction\_mechanism\_classification\_using\_machine\_learning](https://www.researchgate.net/publication/367418407_Organic_reaction_mechanism_classification_using_machine_learning)
34. Machine Learning Materials → Term \- Pollution → Sustainability Directory, 访问时间为 四月 24, 2025， [https://pollution.sustainability-directory.com/term/machine-learning-materials/](https://pollution.sustainability-directory.com/term/machine-learning-materials/)
35. AI/ML in Additive Manufacturing and Polymer Synthesis for New Data and Discovery, 访问时间为 四月 24, 2025， [https://www.chem.uga.edu/events/content/2025/aiml-additive-manufacturing-and-polymer-synthesis-new-data-and-discovery](https://www.chem.uga.edu/events/content/2025/aiml-additive-manufacturing-and-polymer-synthesis-new-data-and-discovery)
36. A Review of the Applications of Machine Learning for Prediction and Analysis of Mechanical Properties and Microstructures in Additive Manufacturing \- ASME Digital Collection, 访问时间为 四月 24, 2025， [https://asmedigitalcollection.asme.org/computingengineering/article/24/12/120801/1206402/A-Review-of-the-Applications-of-Machine-Learning](https://asmedigitalcollection.asme.org/computingengineering/article/24/12/120801/1206402/A-Review-of-the-Applications-of-Machine-Learning)
37. AI-driven, autonomous lab at Argonne transforms materials discovery \- UChicago News, 访问时间为 四月 24, 2025， [https://news.uchicago.edu/story/ai-driven-autonomous-lab-argonne-transforms-materials-discovery](https://news.uchicago.edu/story/ai-driven-autonomous-lab-argonne-transforms-materials-discovery)
38. Autonomous platform for solution processing of electronic polymers \- PMC, 访问时间为 四月 24, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC11833048/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11833048/)
39. Self-driving lab transforms materials discovery \- Argonne National Laboratory, 访问时间为 四月 24, 2025， [https://www.anl.gov/article/selfdriving-lab-transforms-materials-discovery](https://www.anl.gov/article/selfdriving-lab-transforms-materials-discovery)
40. Polybot \- The Center for Nanoscale Materials, 访问时间为 四月 24, 2025， [https://cnm.anl.gov/pages/polybot](https://cnm.anl.gov/pages/polybot)
41. Emerging AI-Driven Materials Technologies Revolutionizing Aerospace, Medical Implants, Optics, and Polymers \- Duke aiM program, 访问时间为 四月 24, 2025， [https://aim-nrt.pratt.duke.edu/news/emerging-ai-driven-materials-technologies-revolutionizing-aerospace-medical-implants-optics](https://aim-nrt.pratt.duke.edu/news/emerging-ai-driven-materials-technologies-revolutionizing-aerospace-medical-implants-optics)
42. MATERIALS – Multiscale Accelerated Research for Innovative Energy and Materials through AI Large-scale Simulations | Argonne National Laboratory, 访问时间为 四月 24, 2025， [https://www.anl.gov/partnerships/materials-multiscale-accelerated-research-for-innovative-energy-and-materials-through-ai-largescale-0](https://www.anl.gov/partnerships/materials-multiscale-accelerated-research-for-innovative-energy-and-materials-through-ai-largescale-0)
43. Accelerating Materials Science with AI and Robotics \- Federation of American Scientists, 访问时间为 四月 24, 2025， [https://fas.org/publication/accelerating-materials-science-with-ai-and-robotics/](https://fas.org/publication/accelerating-materials-science-with-ai-and-robotics/)
44. Prediction and optimization model of sustainable concrete properties using machine learning, deep learning and swarm intelligence: A review | Request PDF \- ResearchGate, 访问时间为 四月 24, 2025， [https://www.researchgate.net/publication/375239857\_Prediction\_and\_optimization\_model\_of\_sustainable\_concrete\_properties\_using\_machine\_learning\_deep\_learning\_and\_swarm\_intelligence\_A\_review](https://www.researchgate.net/publication/375239857_Prediction_and_optimization_model_of_sustainable_concrete_properties_using_machine_learning_deep_learning_and_swarm_intelligence_A_review)
45. The Open Catalyst 2022 (OC22) Dataset and Challenges for Oxide Electrocatalysts | ACS Catalysis \- ACS Publications, 访问时间为 四月 24, 2025， [https://pubs.acs.org/doi/abs/10.1021/acscatal.2c05426](https://pubs.acs.org/doi/abs/10.1021/acscatal.2c05426)
46. PhD Studentship \- Deep Learning-based Mechanistic Elucidation of Processes Mediated by Heterogeneous Catalysts at The University of Manchester \- Jobs.ac.uk, 访问时间为 四月 24, 2025， [https://www.jobs.ac.uk/job/DLN374/phd-studentship-deep-learning-based-mechanistic-elucidation-of-processes-mediated-by-heterogeneous-catalysts](https://www.jobs.ac.uk/job/DLN374/phd-studentship-deep-learning-based-mechanistic-elucidation-of-processes-mediated-by-heterogeneous-catalysts)
47. Research | Internet of Catalysis, 访问时间为 四月 24, 2025， [https://nrt.ku.edu/research](https://nrt.ku.edu/research)
48. Artificial intelligence | University of Chicago News, 访问时间为 四月 24, 2025， [https://news.uchicago.edu/tag/artificial-intelligence](https://news.uchicago.edu/tag/artificial-intelligence)
49. Databases — Introduction to Materials Informatics, 访问时间为 四月 24, 2025， [https://enze-chen.github.io/mi-book-2021/week\_1/02/databases.html](https://enze-chen.github.io/mi-book-2021/week_1/02/databases.html)
50. Materials Project, 访问时间为 四月 24, 2025， [https://next-gen.materialsproject.org/](https://next-gen.materialsproject.org/)
51. NOMAD-Ref Materials Database Overview | Ontosight \- AI Research Assistant, 访问时间为 四月 24, 2025， [https://ontosight.ai/glossary/term/nomad-ref-materials-database-overview--67a174976c3593987a576d63](https://ontosight.ai/glossary/term/nomad-ref-materials-database-overview--67a174976c3593987a576d63)
52. Materials Project API \- Swagger UI, 访问时间为 四月 24, 2025， [https://api.materialsproject.org/docs](https://api.materialsproject.org/docs)
53. Materials API (MAPI) \- Materials Project Documentation, 访问时间为 四月 24, 2025， [https://doc.docs.materialsproject.org/open-apis/the-materials-api/](https://doc.docs.materialsproject.org/open-apis/the-materials-api/)
54. API \- Materials Project, 访问时间为 四月 24, 2025， [https://next-gen.materialsproject.org/api](https://next-gen.materialsproject.org/api)
55. Aflow \- Automatic FLOW for Materials Discovery, 访问时间为 四月 24, 2025， [https://aflowlib.org/](https://aflowlib.org/)
56. Automatic \- FLOW for Materials Discovery \- AFlow, 访问时间为 四月 24, 2025， [https://aflow.dev.materials.duke.edu/documentation](https://aflow.dev.materials.duke.edu/documentation)
57. AFLOW Database and APIs: AFLOW.org, AFLUX, AFLOW REST-API, 访问时间为 四月 24, 2025， [https://aflow.dev.materials.duke.edu/aflow-school/past\_schools/20210906/08\_aflow\_school\_database\_aflux.pdf](https://aflow.dev.materials.duke.edu/aflow-school/past_schools/20210906/08_aflow_school_database_aflux.pdf)
58. AFLOW Database Entries — aflow 0.0.7 documentation \- GitHub Pages, 访问时间为 四月 24, 2025， [https://rosenbrockc.github.io/aflow/entries.html](https://rosenbrockc.github.io/aflow/entries.html)
59. (PDF) The Open Quantum Materials Database (OQMD): Assessing the accuracy of DFT formation energies \- ResearchGate, 访问时间为 四月 24, 2025， [https://www.researchgate.net/publication/286490327\_The\_Open\_Quantum\_Materials\_Database\_OQMD\_Assessing\_the\_accuracy\_of\_DFT\_formation\_energies](https://www.researchgate.net/publication/286490327_The_Open_Quantum_Materials_Database_OQMD_Assessing_the_accuracy_of_DFT_formation_energies)
60. OQMD RESTful API — qmpy v1.2.0 documentation, 访问时间为 四月 24, 2025， [https://static.oqmd.org/static/docs/restful.html](https://static.oqmd.org/static/docs/restful.html)
61. OQMD Toolkits, 访问时间为 四月 24, 2025， [https://mohanliu.github.io/oqmddoc/](https://mohanliu.github.io/oqmddoc/)
62. oqmd-v1.2-dataset-for-cgnn/OQMD\_v1\_2\_dataset\_for\_CGNN.ipynb at main · Tony-Y/oqmd-v1.2-dataset-for-cgnn · GitHub, 访问时间为 四月 24, 2025， [https://github.com/Tony-Y/oqmd-v1.2-dataset-for-cgnn/blob/main/OQMD\_v1\_2\_dataset\_for\_CGNN.ipynb](https://github.com/Tony-Y/oqmd-v1.2-dataset-for-cgnn/blob/main/OQMD_v1_2_dataset_for_CGNN.ipynb)
63. Materials Cloud, 访问时间为 四月 24, 2025， [https://www.materialscloud.org/home](https://www.materialscloud.org/home)
64. Materials Data Facility (MDF) Portal \- Globus, 访问时间为 四月 24, 2025， [https://www.globus.org/user-stories/materials-data-facility-portal](https://www.globus.org/user-stories/materials-data-facility-portal)
65. Accelerating materials discovery and design with machine learning \- LUMI supercomputer, 访问时间为 四月 24, 2025， [https://lumi-supercomputer.eu/accelerating-materials-discovery-and-design-with-machine-learning/](https://lumi-supercomputer.eu/accelerating-materials-discovery-and-design-with-machine-learning/)
66. Graph Neural Networks Based Deep Learning for Predicting Structural and Electronic Properties \- arXiv, 访问时间为 四月 24, 2025， [https://arxiv.org/html/2411.02331v1](https://arxiv.org/html/2411.02331v1)
67. \[1710.10324\] Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties \- arXiv, 访问时间为 四月 24, 2025， [https://arxiv.org/abs/1710.10324](https://arxiv.org/abs/1710.10324)
68. Advancements in Molecular Property Prediction: A Survey of Single and Multimodal Approaches \- arXiv, 访问时间为 四月 24, 2025， [https://arxiv.org/html/2408.09461v2](https://arxiv.org/html/2408.09461v2)
69. Enhancing material property prediction with ensemble deep graph convolutional networks, 访问时间为 四月 24, 2025， [https://arxiv.org/html/2407.18847v1](https://arxiv.org/html/2407.18847v1)
70. A Transformer Model for Predicting Generic Chemical Reaction Products from Templates \- arXiv, 访问时间为 四月 24, 2025， [https://www.arxiv.org/pdf/2503.05810](https://www.arxiv.org/pdf/2503.05810)
71. Transformers for molecular property prediction: Lessons learned from the past five years \- arXiv, 访问时间为 四月 24, 2025， [https://arxiv.org/pdf/2404.03969?](https://arxiv.org/pdf/2404.03969)  
72. \[PDF\] Materials Informatics Transformer: A Language Model for Interpretable Materials Properties Prediction | Semantic Scholar, 访问时间为 四月 24, 2025， [https://www.semanticscholar.org/paper/0d5a69f00d52dd2013d270d34e3b1dee2a54d286](https://www.semanticscholar.org/paper/0d5a69f00d52dd2013d270d34e3b1dee2a54d286)
73. \[2303.12188\] Toward Accurate Interpretable Predictions of Materials Properties within Transformer Language Models \- arXiv, 访问时间为 四月 24, 2025， [https://arxiv.org/abs/2303.12188](https://arxiv.org/abs/2303.12188)
74. Data quantity governance for machine learning in materials science \- Oxford Academic, 访问时间为 四月 24, 2025， [https://academic.oup.com/nsr/article/10/7/nwad125/7147579](https://academic.oup.com/nsr/article/10/7/nwad125/7147579)
75. \[2208.09481\] Graph neural networks for materials science and chemistry \- arXiv, 访问时间为 四月 24, 2025， [https://arxiv.org/abs/2208.09481](https://arxiv.org/abs/2208.09481)
76. Machine learning of material properties: Predictive and interpretable multilinear models, 访问时间为 四月 24, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC9075804/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9075804/)
77. Addressing the Data Scarcity Problem in Ecotoxicology via Small Data Machine Learning Methods | Environmental Science & Technology \- ACS Publications, 访问时间为 四月 24, 2025， [https://pubs.acs.org/doi/10.1021/acs.est.5c00510](https://pubs.acs.org/doi/10.1021/acs.est.5c00510)
78. SynCoTrain: A Dual Classifier PU-learning Framework for Synthesizability Prediction \- arXiv, 访问时间为 四月 24, 2025， [https://arxiv.org/html/2411.12011v1](https://arxiv.org/html/2411.12011v1)
79. \[2411.12011\] SynCoTrain: A Dual Classifier PU-learning Framework for Synthesizability Prediction \- arXiv, 访问时间为 四月 24, 2025， [https://arxiv.org/abs/2411.12011](https://arxiv.org/abs/2411.12011)
80. \[2109.12283\] Scalable deeper graph neural networks for high-performance materials property prediction \- arXiv, 访问时间为 四月 24, 2025， [https://arxiv.org/abs/2109.12283](https://arxiv.org/abs/2109.12283)
81. Derivative-based pre-training of graph neural networks for materials property predictions, 访问时间为 四月 24, 2025， [https://pubs.rsc.org/en/content/articlehtml/2024/dd/d3dd00214d](https://pubs.rsc.org/en/content/articlehtml/2024/dd/d3dd00214d)
82. Using GNN property predictors as molecule generators \- arXiv, 访问时间为 四月 24, 2025， [https://arxiv.org/html/2406.03278v1](https://arxiv.org/html/2406.03278v1)
83. \[D\] Why I'm Lukewarm on Graph Neural Networks : r/MachineLearning \- Reddit, 访问时间为 四月 24, 2025， [https://www.reddit.com/r/MachineLearning/comments/kqazpd/d\_why\_im\_lukewarm\_on\_graph\_neural\_networks/](https://www.reddit.com/r/MachineLearning/comments/kqazpd/d_why_im_lukewarm_on_graph_neural_networks/)
84. Machine Learning for Materials Discovery (ML4MD)Machine Learning for Materials Discovery (ML4MD) \- CECAM, 访问时间为 四月 24, 2025， [https://www.cecam.org/workshop-details/machine-learning-for-materials-discovery-ml4md-1417](https://www.cecam.org/workshop-details/machine-learning-for-materials-discovery-ml4md-1417)
85. Opportunities and challenges of diffusion models for generative AI \- Oxford Academic, 访问时间为 四月 24, 2025， [https://academic.oup.com/nsr/article/11/12/nwae348/7810289](https://academic.oup.com/nsr/article/11/12/nwae348/7810289)
86. Task-Specific Transformer-Based Language Models in Health Care: Scoping Review, 访问时间为 四月 24, 2025， [https://medinform.jmir.org/2024/1/e49724](https://medinform.jmir.org/2024/1/e49724)
87. Clinical concept extraction using transformers \- PMC \- PubMed Central, 访问时间为 四月 24, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC7727351/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7727351/)
88. Materials Informatics Transformer: A Language Model for Interpretable Materials Properties Prediction | Request PDF \- ResearchGate, 访问时间为 四月 24, 2025， [https://www.researchgate.net/publication/373551771\_Materials\_Informatics\_Transformer\_A\_Language\_Model\_for\_Interpretable\_Materials\_Properties\_Prediction](https://www.researchgate.net/publication/373551771_Materials_Informatics_Transformer_A_Language_Model_for_Interpretable_Materials_Properties_Prediction)
89. \[2308.16259\] Materials Informatics Transformer: A Language Model for Interpretable Materials Properties Prediction \- arXiv, 访问时间为 四月 24, 2025， [https://arxiv.org/abs/2308.16259](https://arxiv.org/abs/2308.16259)
90. The JARVIS Infrastructure is All You Need for Materials Design \- arXiv, 访问时间为 四月 24, 2025， [https://arxiv.org/html/2503.04133v1](https://arxiv.org/html/2503.04133v1)
91. Evaluation Metrics for Machine Learning Models: Part 1 \- Paperspace Blog, 访问时间为 四月 24, 2025， [https://blog.paperspace.com/ml-evaluation-metrics-part-1/](https://blog.paperspace.com/ml-evaluation-metrics-part-1/)
92. Know The Best Evaluation Metrics for Your Regression Model \- Analytics Vidhya, 访问时间为 四月 24, 2025， [https://www.analyticsvidhya.com/blog/2021/05/know-the-best-evaluation-metrics-for-your-regression-model/](https://www.analyticsvidhya.com/blog/2021/05/know-the-best-evaluation-metrics-for-your-regression-model/)
93. evaluation metrics of MSE,MAE and RMSE \- Stack Overflow, 访问时间为 四月 24, 2025， [https://stackoverflow.com/questions/78154131/evaluation-metrics-of-mse-mae-and-rmse](https://stackoverflow.com/questions/78154131/evaluation-metrics-of-mse-mae-and-rmse)
94. A Comprehensive Overview of Regression Evaluation Metrics | NVIDIA Technical Blog, 访问时间为 四月 24, 2025， [https://developer.nvidia.com/blog/a-comprehensive-overview-of-regression-evaluation-metrics/](https://developer.nvidia.com/blog/a-comprehensive-overview-of-regression-evaluation-metrics/)
95. Choosing right metrics for regression model \- Stack Overflow, 访问时间为 四月 24, 2025， [https://stackoverflow.com/questions/60869083/choosing-right-metrics-for-regression-model](https://stackoverflow.com/questions/60869083/choosing-right-metrics-for-regression-model)
96. Study on Accuracy Metrics for Evaluating the Predictions of Damage Locations in Deep Piles Using Artificial Neural Networks with Acoustic Emission Data \- MDPI, 访问时间为 四月 24, 2025， [https://www.mdpi.com/2076-3417/11/5/2314](https://www.mdpi.com/2076-3417/11/5/2314)
97. (PDF) The coefficient of determination R-squared is more informative than SMAPE, MAE, MAPE, MSE and RMSE in regression analysis evaluation \- ResearchGate, 访问时间为 四月 24, 2025， [https://www.researchgate.net/publication/353007143\_The\_coefficient\_of\_determination\_R-squared\_is\_more\_informative\_than\_SMAPE\_MAE\_MAPE\_MSE\_and\_RMSE\_in\_regression\_analysis\_evaluation](https://www.researchgate.net/publication/353007143_The_coefficient_of_determination_R-squared_is_more_informative_than_SMAPE_MAE_MAPE_MSE_and_RMSE_in_regression_analysis_evaluation)
98. Machine learning benchmarks: MAE, RMSE, and R-squared \- Cross Validated, 访问时间为 四月 24, 2025， [https://stats.stackexchange.com/questions/645848/machine-learning-benchmarks-mae-rmse-and-r-squared](https://stats.stackexchange.com/questions/645848/machine-learning-benchmarks-mae-rmse-and-r-squared)
99. A Higher r-squared always implies a reduction in MAE and RMSE? \- Cross Validated, 访问时间为 四月 24, 2025， [https://stats.stackexchange.com/questions/592411/a-higher-r-squared-always-implies-a-reduction-in-mae-and-rmse](https://stats.stackexchange.com/questions/592411/a-higher-r-squared-always-implies-a-reduction-in-mae-and-rmse)
100. Metrics for Linear Regression Effectiveness: R-squared, MSE and RSE | Saylor Academy, 访问时间为 四月 24, 2025， [https://learn.saylor.org/mod/page/view.php?id=80811](https://learn.saylor.org/mod/page/view.php?id=80811)
101. Top 10 Challenges in Artificial Intelligence for Materials and ..., 访问时间为 四月 24, 2025， [https://citrine.io/top-10-challenges-in-artificial-intelligence-for-materials-and-chemicals/](https://citrine.io/top-10-challenges-in-artificial-intelligence-for-materials-and-chemicals/)
102. \[2207.13880\] Towards overcoming data scarcity in materials science: unifying models and datasets with a mixture of experts framework \- arXiv, 访问时间为 四月 24, 2025， [https://arxiv.org/abs/2207.13880](https://arxiv.org/abs/2207.13880)
103. Materials Informatics: The AI-Designed Materials Revolution | IDTechEx Research Article, 访问时间为 四月 24, 2025， [https://www.idtechex.com/en/research-article/materials-informatics-the-ai-designed-materials-revolution/30643](https://www.idtechex.com/en/research-article/materials-informatics-the-ai-designed-materials-revolution/30643)
104. Effects of Machine Learning Algorithms for Predicting and Optimizing the Properties of New Materials in the United States \- AJPO Journals, 访问时间为 四月 24, 2025， [https://ajpojournals.org/journals/index.php/EJPS/article/download/1444/1558](https://ajpojournals.org/journals/index.php/EJPS/article/download/1444/1558)
105. The future of self-driving laboratories: from human in the loop interactive AI to gamification, 访问时间为 四月 24, 2025， [https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00040d](https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00040d)
106. Autonomous materials discovery and manufacturing (AMDM): A review and perspectives, 访问时间为 四月 24, 2025， [https://par.nsf.gov/servlets/purl/10488816](https://par.nsf.gov/servlets/purl/10488816)
107. Self-Driving Labs: AI and Robotics Accelerating Materials Innovation \- CSIS, 访问时间为 四月 24, 2025， [https://www.csis.org/blogs/perspectives-innovation/self-driving-labs-ai-and-robotics-accelerating-materials-innovation](https://www.csis.org/blogs/perspectives-innovation/self-driving-labs-ai-and-robotics-accelerating-materials-innovation)
108. AI-Powered “Self-Driving” Labs: Accelerating Life Science R\&D | Tips and Tricks \- Scispot, 访问时间为 四月 24, 2025， [https://www.scispot.com/blog/ai-powered-self-driving-labs-accelerating-life-science-r-d](https://www.scispot.com/blog/ai-powered-self-driving-labs-accelerating-life-science-r-d)
109. Self-Driving Laboratories for Chemistry and Materials Science \- PMC \- PubMed Central, 访问时间为 四月 24, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC11363023/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11363023/)
