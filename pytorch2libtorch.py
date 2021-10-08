import torch
import torch.nn as nn
from model import PConvNet

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

pconvnet = PConvNet().cuda()
state_dict = torch.load('./weights/checkpoint_mask_lightest_16.8.pth')
new_state_dict = {}
for k in state_dict.keys():
    new_k = k[7:]
    new_state_dict[new_k] = state_dict[k]
pconvnet.load_state_dict(new_state_dict,strict=True)
pconvnet.eval()

x = torch.rand(1,3,256,256).cuda()
mask = torch.rand(1,3,256,256).cuda()
with torch.jit.optimized_execution(True):
    traced_model = torch.jit.trace(pconvnet,(x,mask))
    traced_model.save('./libtorch model/partialconv.pt')
y = traced_model(pconvnet,(torch.rand(1,3,256,256).cuda(),torch.rand(1,3,256,256).cuda()))
print(y)


'''
1. partial convolution在论文里面表现为将X乘一个mask，然后再乘个**mask里1的个数/mask的元素个数**做平均

2. 英伟达提出了个`irregular mask dataset`

3. loss函数由很多部分组成，其中style loss和perceptual loss是很重要的

4. 数据集是ImageNet，Places2，celeba

5. 向人群做了A/B test

6. 还可以用来干超分辨率，效果比SRGAN好

7. numpy生成01矩阵

   ```python
   np.random.randint(0,2,(h,w))
   ```

8. torch生成01矩阵

   ```python
   torch.randint(0,2,(h,w)) # 和numpy一样都是左闭右开
   ```

9. torch创建tensor

   ```python
   torch.tensor()
   torch.FloatTensor() # 创建float32
   ```

10. torch的卷积H和W是向下取整的

11. torch的UpSample是直接乘以scale_factor然后向下取整的

12. encode的部分stride全是2，decode的全是1

13. 不知道为什么，不用Sequential好像不能cuda

14. Pconv的初始化方式是xavier，但是又有个kaiming，先改一下成kaiming

15. torch的tensor可以直接1减

16. 模型在`pytorch/vision/torchvision/models`里面

17. 损失函数里面的comp，其实是`mask * input + (1 - mask) * output`，with the non-hole pixels directly set to ground truth；也就是用pred的非孔像素直接设置给原来gt的变白了的像素（孔像素）

18. 因为perceptual和style loss都需要vgg16的分离层，所以写在一起

19. vgg16 extractor是抽取**三个位置**，[0,5),[5,10),[10,17),然后分别输入y_pred、y_comp和gt，各自得到三个feature map（pred三个，comp三个，gt三个），然后将pred和comp的feature map都和gt对应的位置作loss，原论文是每一层都做一次loss，官方代码也是只用了三个层的输出；

20. perceptual和style loss有一个版本有normalize，有一个没有，**论文没有但是官方代码有**，按官方的来

21. nn.L1loss()默认的reduction参数是‘mean’

22. gram matrix用做风格迁移的loss，是一组向量的每个元素和另外元素的内积，如果是矩阵的话，先将矩阵打平，然后前面是n×n行1列，后面是一行n×n列，然后最终得到(n×n)^2的gram matrix，**但是通常是三维的(c,h,w)的求gram matrix，所以其实是将h和w乘起来，变成(c,h*w) * (h*w,c)，所以可以认为是i通道的feature map和j通道的feature map的互相关矩阵，**就哪两个特征是相辅相成哪两个是此消彼长，最终在保证内容的情况下，进行风格的传输

    > 于是，在一个Gram矩阵中，既能体现出有哪些特征，又能体现出不同特征间的紧密程度。论文中作者把这个定义为风格

23. 论文里面的style loss有除以系数，而且因为是除以chw，不能用torch.mean

24. tv loss里面的Ii,j+1 - Ii,j其实就是后一个元素减前一个，取绝对值可以搞定，官方的实现很简单

25. mask是自己生成出来的，而且在最后就是乘而已

26. https://nv-adlr.github.io/publication/partialconv-inpainting partial convolution的M是(c,h,w)都与输入图像相同的，所以其实mask是个卷积的结果，因为用相同的channel和kernel，能保证mask的chw与图像相同，然后就是直接乘

27. torch.clamp(x,min=...,max=...)类似np.clip()，将小于min的元素变成min，大于max的变成max

28. 最大的疑惑点是，为什么要把mask进行卷积；相比源代码，把mask的卷积变成了`with torch.no_grad()`，并且直接返回mask

29. opencv旋转后默认是变黑的，如果要指定颜色或者插值的话，需要指定borderValue或者borderMode,https://www.jianshu.com/p/fef3733e1183

30. opencv的dilate和erode实现膨胀和腐蚀，都是背景膨胀或腐蚀（具体表现就是图形变小和变大），然后可以将图像分开或者实现其他效果

    > 腐蚀：腐蚀会把物体的边界腐蚀掉，卷积核沿着图象滑动，如果卷积核对应的原图的所有像素值为1，那么中心元素就保持原来的值，否则变为零。主要应用在去除白噪声，也可以断开连在一起的物体。
    >
    > 膨胀：卷积核所对应的原图像的像素值只要有一个是1，中心像素值就是1。一般在除噪是，先腐蚀再膨胀，因为腐蚀在去除白噪声的时候也会使图像缩小，所以我们之后要进行膨胀。当然也可以用来将两者物体分开。
    >
    > 开运算和闭运算就是将腐蚀和膨胀按照一定的次序进行处理。
    > 但这两者并不是可逆的，即先开后闭并不能得到原先的图像。
    > 为了获取图像中的主要对象：对一副二值图连续使用闭运算和开运算，或者消除图像中的噪声，也可以对图像先用开运算后用闭运算，不过这样也会消除一些破碎的对象。
    >
    > 开运算：先腐蚀后膨胀，用于移除由图像噪音形成的斑点。
    > 闭运算：先膨胀后腐蚀，用来连接被误分为许多小块的对象；

    其中卷积的kernel可以直接用np.ones生成，也可以`cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size,kernel_size))`来生成

31. 其实将mask经过卷积是有道理的，不仅可以保证长宽通道都与图片相同，还能保证经过了同样的操作，保证了一致性，本身随机生成的pair是没有规律的

32. mask和img其实都是要transform的，同样就行

33. contract那里的stride全是2，extract那里的stride全是1，upsample全是2

34. Partial Convolution里面的weight_mask是torch.ones()出来的，需要cuda()

35. ```python
    ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 512, 1, 1]) # batchnorm要每个通道的值个数大于1
    ```

36. mask也要upsample和concat

37. ```python
    ValueError: assignment destination is read-only
    ```

    这个错，要将mask变成numpy的array，然后copy才行，差一个都不行

    ```python
    mask = np.asarray(mask).copy()
    ```

38. 网络输出的图像是需要denormalize的，不然归一化的很难看

39. finetune的权重里面，pconv是没有bias的，bn是有bias的

40. 有bias比没有bias下降慢，没有bias好

41. 最后一层没有leaky_relu，有没有bn

42. mask不要normalize

43. vgg16是用没有bn的版本

44. mask不要用白色的了，因为y_comp是`y_pred * (1 - mask) + y_true * mask`，因为原话是：

    > Icomp is the raw output image Iout, but with the non-hole pixels directly set to ground truth;

    所以是将y_pred被mask遮掉意外的预测，设置到真实图片上，但是如果是255就会出问题，白色再乘上input的话就不对了

45. 官方的vgg16 extractor是没有Normalization层的

# 46.大坑！

生成mask那里，必须得是255的图像保存下来，不能是1的图像保存下来，因为之后ToTensor()那里会除以255，所以如果是全0和1的话，就会变成0.0039，到时候y_comp就会出问题，所以必须在这里变成255的先

因为y_comp = mask * y_true + (1 - mask) * y_pred，mask是0.0039，所以y_pred就会占很大的成分，导致y_comp看起来跟y_pred一样

所以必须在ToTensor之后相乘，不能之前相乘

47. torch可以ToPILImage()然后显示，ToTensor之后不能fromarray变成PIL图像

48. PIL.Image.convert(‘1’)是变成二值图像，非黑即白，0~255，‘L’是灰度图，灰度按这个公式计算

    ```python
    L = R * 299/1000 + G * 587/1000+ B * 114/1000
    ```

49. loss那里的feature_extraction将Normalization层去掉了，效果更好

50. batch_size是32好像效果没有8好

51. libtorch加载模型，据说是1.1以下是：

    ```c++
    std::shared_ptr<torch::jit::script::Module> pconvnet = torch::jit::load("../model/partialconv.pt");
    assert(pconvnet != nullptr);
    pconvnet->to(at::kCUDA);
    ```

    1.2以上是：

    ```c++
    torch::jit::script::Module pconvnet = torch::jit::load("../model/partialconv.pt");
    pconvnet.to(at::kCUDA);
    ```

    

'''