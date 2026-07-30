给你的任务：

合并仓库，把PaddleFormers和PaddleFleet合为一个仓库，重新命名为PaddleFleet，PaddleFormers代码路径在/root/paddlejob/share-storage/gpfs/system-public/liujiaodi/PaddleFormers


合并原则：

PaddleFormers主要分为docs，examples，paddleformers，scripts，skills,tests,third_party几个目录,针对几个目录的处理原则如下：
（1）paddleformers目录迁移到src/paddlefleet目录，所有的命名空间由paddleformers.xxx改成paddlefleet.xxx，有一个目录generation和PaddleFleet仓库目录重复了，得把这些文件合二为一，取并集；
（2）如果迁移过程发现有一些文件名重复了，你也可以抛出来和我讨论，确定后思路再接着干；（2）third_party不管，因为PaddleFormers的third_party实际上没有用到，全部删掉就好；
（3）tests目录迁移到test/formers/目录下，可以新建这个test/formers目录；
（4）srcipts目录的文件就迁移到scripts目录；
（5）skills和exaples目录直接复制过来就好；
（6）原先PaddleFormers的仓库还有setup.py，我们在新仓库是不需要的，PaddleFleet的仓库是用的uv workspace的逻辑，PaddleFormers是纯python代码库，迁移过来全部编译到paddlefleet纯python包里，然后不再维护requremens.txt和setup.py,PaddleFormers的requirements.txt都是运行时需要的，把这些依赖都放到paddlefleet的依赖里
迁移完之后原先所有的paddleformers.xxx都变成paddlefleet.xxx，


其他细节：

其他细节我暂时没想到，你迁移的时候遇到一些问题就可以立马和我讨论。


CI如何迁移？

PaddleFormers的CI迁移到PaddleFleet，只用关心两条CI，其他的都不需要，Model Unittest GPU CI和Model Unittest GPU CI，放到Test-release.yml里，依赖Build Fleet whl



这些细节你也需要处理下：

（1）.github/workflows/Test-release.yml里原先需要git clone -b develop https://github.com/PaddlePaddle/PaddleFormers.git，被你改成了git clone PaddleFleet了，好需要Clone吗？根本不需要啊，你参考下Multi-card test，Install PaddleFleet里不是wget了paddlefleet.tar，gz包吗？你得思考一下，不要机械的替换