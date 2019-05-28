btb

![1558958606591](Branch_Prediction.assets/1558958606591.png)

bht

![1558958856543](Branch_Prediction.assets/1558958856543.png)



```
t0: x5
t1: x6
t2: x7
t3: x28
t4: x29
t5: x30
t6: x31
```







| BTB  | BHT  | Real | NPC_Pred | flush | NPC_Real | BTB update |
| ---- | ---- | ---- | -------- | ----- | -------- | ---------- |
| Y    | Y    | Y    | BUF      | N     | BrNPC    | N          |
| Y    | Y    | N    | BUF      | Y     | PC_EX+4  | Y          |
| Y    | N    | Y    | PC_IF+4  | Y     | BrNPC    | Y          |
| Y    | N    | N    | PC_IF+4  | N     | PC_IF+4  | N          |
| N    | Y    | Y    | PC_IF+4  | Y     | BrNPC    | Y          |
| N    | Y    | N    | PC_IF+4  | N     | PC_IF+4  | N          |
| N    | N    | Y    | PC_IF+4  | Y     | BrNPC    | Y          |
| N    | N    | N    | PC_IF+4  | N     | PC_IF+4  | N          |

