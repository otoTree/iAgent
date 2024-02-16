

# iAgent 探索 AGI 的可能性

### IAgent explores the possibility of AGI



​       我们在设计中大量运用function call的思想，由大模型驱动各类工具，因此我们在进行各类function call之前设置一个触发器，由大模型自行判断是否生成新的status，否则延续上一status，本次拒绝生成新的status只会记录在触发器内部

​      We extensively utilize the concept of function calls in our design, driven by the large model that drives various tools. Therefore, before conducting various function calls, we set a trigger for the large model to determine whether to generate a new status. Otherwise, we will continue with the previous status. Refusing to generate a new status this time will only be recorded inside the trigger

![AGENT](C:\Users\86134\Documents\AGENT(1).png)



