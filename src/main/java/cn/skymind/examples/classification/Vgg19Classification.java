

package cn.skymind.examples.classification;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.zoo.model.VGG19;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;


/**
 * FashionMnist分类
 * @author wenfengxu
 */
@Slf4j
public class Vgg19Classification {

    protected static int height = 224;

    protected static int width = 224;

    protected static int channels =3;

    protected static int batchSize = 96;

    protected static long seed = 42;

    protected static Random random = new Random(seed);

    protected static int epochs = 1;

    public static String dataSetPath ="dataset/Fashion-MNIST";

    private static double learningRate=0.001;

    public  static void train() throws Exception {

        File mainPath = new File(dataSetPath);

        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, random);
        //获取分类标签数
        int numLabels = Objects.requireNonNull(fileSplit.getRootDir().listFiles(File::isDirectory)).length;

        //父路径标签创建器，把训练数据所在的父目录作为分类标签
        ParentPathLabelGenerator parentPathLabelGenerator = new ParentPathLabelGenerator();
        /**用一个BalancedPathFilter来抽样，来实现样本均衡，提高模型性能
         需要注意的是，BalancedPathFilter抽样出来的总数量=最少样本的标签对应的样本数*标签数
         例如，有4个文件夹a,b,c,d对应的样本数量为5,10,15,20使用BalancedPathFilter之后
         抽出来的样本总数量=5*4=20个，而不是5+10+15+20=50个
         **/
        BalancedPathFilter balancedPathFilter = new BalancedPathFilter(random, null, parentPathLabelGenerator);
        //训练与测试的数据分割比例，适用于训练与测试数据没有分开的情况
        double splitTrainTest = 0.8;
        //输入分割器
        InputSplit[] inputSplit = fileSplit.sample(balancedPathFilter, splitTrainTest, 1 - splitTrainTest);
        //训练集输入分割器
        InputSplit trainInputSplit = inputSplit[0];
        //测试集输入分割器
        InputSplit testInputSplit = inputSplit[1];
        //创建归一化器，用于将图片数据归一化
        DataNormalization dataNormalization = new ImagePreProcessingScaler(0, 1);

        //图片读取器
        ImageRecordReader trainImageRecordReader = new ImageRecordReader(height, width, channels, parentPathLabelGenerator);
        //初始化读取器
        trainImageRecordReader.initialize(trainInputSplit, null);
        //创建训练数据集迭代器
        DataSetIterator trainDataSetIterator = new RecordReaderDataSetIterator(trainImageRecordReader, batchSize, 1, numLabels);
        //数据归一化
        dataNormalization.fit(trainDataSetIterator);
        //数据归一化
        trainDataSetIterator.setPreProcessor(dataNormalization);

        //训练集
        ImageRecordReader testImageRecordReader = new ImageRecordReader(height, width, channels, parentPathLabelGenerator);

        testImageRecordReader.initialize(testInputSplit);

        DataSetIterator testDataSetIterator = new RecordReaderDataSetIterator(testImageRecordReader, batchSize, 1, numLabels);

        dataNormalization.fit(testDataSetIterator);

        testDataSetIterator.setPreProcessor(dataNormalization);

        ComputationGraph network= buildNetwork(numLabels);

        //训练UI
        UIServer uiServer = UIServer.getInstance();
        //UI数据保存方式为文件保存
        StatsStorage statsStorage = new InMemoryStatsStorage();

        uiServer.attach(statsStorage);
        //设置训练网络监听器，打印损失得分，并且每200次迭代进行一次评估
        network.setListeners(new StatsListener( statsStorage), new ScoreIterationListener(1), new EvaluativeListener(testDataSetIterator, 200, InvocationType.ITERATION_END));
        //训练模型
        network.fit(trainDataSetIterator, epochs);
        //图片转换实现数据增强
        ImageTransform flipTransform1 = new FlipImageTransform(random);

        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));

        ImageTransform warpTransform = new WarpImageTransform(random, 42);

        boolean shuffle = false;

        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(new Pair<>(flipTransform1,0.9),
                new Pair<>(flipTransform2,0.8),
                new Pair<>(warpTransform,0.5));

        //noinspection ConstantConditions
        ImageTransform transform = new PipelineImageTransform(pipeline,shuffle);

        //使用数据增强进行训练
        trainImageRecordReader.initialize(trainInputSplit, transform);

        trainDataSetIterator = new RecordReaderDataSetIterator(trainImageRecordReader, batchSize, 1, numLabels);

        dataNormalization.fit(trainDataSetIterator);

        trainDataSetIterator.setPreProcessor(dataNormalization);
        //训练网络
        network.fit(trainDataSetIterator, epochs);

        log.info("****************示例结束********************");
    }








    public static ComputationGraph buildNetwork(int numLabels){

       return  VGG19.builder().numClasses(numLabels).build().init();

    }






    public static void main(String[] args) throws Exception {
        train();
    }

}
