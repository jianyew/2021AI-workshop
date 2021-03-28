

# Sagemaker



•Amazon SageMaker有助于降低机器学习实现成本：以往的数据标记工作往往成本高昂，而且为了保证准确性，只能高度依赖人工标记。Amazon SageMaker Ground Truth可帮助用户自动获取数据的“切实真相”，在不牺牲准确性的前提下将规模化标记成本削减达70%。

•弹性至关重要。模型对于资源的需求往往难以预定确定，而且可能随着新数据的引入以及训练算法的更新而发生大幅改变。有鉴于此，以往的资源与成本权衡通常取决于操作者的猜测甚至直觉，而且当前判断可能很快过时。Amazon Elastic Inference能够帮助用户快速配置ML工作负载所需要的适量GPU资源。

•用户可以利用Managed Spot Training将机器学习模型的训练成本降低达90%。Managed Spot Training利用Amazon EC2竞价实例，这部分实例资源为EC2服务的备用容量。因此与Amazon EC2按需实例相比，模型训练任务的运行成本得到大幅降低。Amazon SageMaker负责训练任务的管理，保证任务只在备用算力充裕时运行。以此为基础，用户不必持续轮询容量水平；此外，Managed Spot Training全面接管训练流程，消除了因新工具介入导致的流程中断。Managed Spot Training还支持Amazon SageMaker提供的自动模型调优、内置算法与框架，同时兼容其他多种自定义算法。

•在三年期比较周期之内，相较于其他云ML解决方案（例如自托管Amazon EC2以及AWS托管Amazon EKS），Amazon SageMaker能够将总体拥有成本（TCO）降低至少54%。统计数据来自一支由5名数据科学家组建的小型团队，以及一支包含250位数据科学家成员的超大规模团队。除了显著降低TCO之外，Amazon SageMaker丰富的生产力功能还帮助用户更快将机器学习原型设计投入生产，使数据科学家的生产效率提高达10倍。



规模化与性能

•AWS的TensorFlow优化成果可跨越数百个GPU提供近线性扩展效率，可立足云环境运行，确保用户轻松便捷地训练出更准确、更复杂的机器学习模型。

•利用Amazon SageMaker Neo，用户可以持续优化模型以实现2倍性能提升，确保您的推理（预测）模型处于最佳性能水平。

•Amazon SageMaker内置安全保护机制。我们为用户提供：端到端加密、专用网络连接、授权、日志记录与可审计性、安全规则保障以及notebook生命周期配置。



易于使用

•用户可以通过全球首款面向机器学习的IDE（Amazon SageMaker Studio）访问ML模型构建、训练及部署中所需要用到的全部组件。

•Amazon SageMaker极大降低了机器学习的复杂性——开发人员只需在SageMaker控制台内点击一下，即可轻松实现模型的获取与部署，同时保证一次训练、随处运行（从云端到边缘位置）。





![image-20210329003942485](/Users/wjianye/kspace/客户材料/TCL/TCL客户项目/2021.04 Aiworkshop/2021AI-workshop/images/image-20210329003942485.png)