# A*

## 地图

### 小地图

每次实验跑5次观察结果

#### 少障碍物：



![image-20250808221839598](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808221839598.png)

每次的路径都是一致，且为最优，性能较好

#### 多障碍物：



![image-20250808221908276](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808221908276.png)

![image-20250808222446781](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808222446781.png)

性能下降严重

### 大地图

#### 少障碍物

（4,4）到（160,80）

![image-20250808222749166](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808222749166.png)





（4,4）到（120,60）

![image-20250808222826837](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808222826837.png)

#### 多障碍物

（1,1）到（180,100）

![image-20250808223241120](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808223241120.png)

![image-20250808223507524](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808223507524.png)

# RRT

## 地图

### 小地图

和步长，概率，迭代次数关系巨大



#### 少障碍物

（2,2）到（49,25）

第一次：

![image-20250808224100488](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808224100488.png)

第二次：

![image-20250808225311555](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808225311555.png)

第三次：

![image-20250808225856894](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808225856894.png)

第四次：

![image-20250808225912356](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808225912356.png)

第五次：

![image-20250808225924016](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808225924016.png)

#### 多障碍物

（2,2）到（49,25）

第一次：

![image-20250808230252596](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808230252596.png)



第二次：

```python
No Path Found!
```

第三次

No Path Found!

第四次：

![](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808230359030.png)

第五次：

![image-20250808230442447](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808230442447.png)

### 大地图

#### 少障碍物

（2,2）到（160,80）   0.5，0.1,1000000

第一次：

![image-20250808231658274](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808231658274.png)

第二次：

![image-20250808231710174](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808231710174.png)

第三次：

![image-20250808231725600](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808231725600.png)

第四次：

![image-20250808231735931](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808231735931.png)

第五次：

![image-20250808231748724](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808231748724.png)



将步长减小到0.1：

![image-20250808232216644](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808232216644.png)

![image-20250808231841831](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808231841831.png)



把目标点概率调大到0.5：

![image-20250808231855419](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808231855419.png)

理论上能降低，但也有可能依然复杂

![image-20250808232809073](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808232809073.png)



#### 多障碍物

第一次：

![image-20250808233914151](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808233914151.png)

第二次：

![image-20250808233932384](C:\Users\11618\AppData\Roaming\Typora\typora-user-images\image-20250808233932384.png)