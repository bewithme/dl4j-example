package cn.skymind.examples.objectdetection;


import cn.skymind.examples.utils.JsonUtils;
import cn.skymind.examples.utils.ModelTrainOptions;
import com.beust.jcommander.JCommander;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.*;


/**
 * Yolo2模型训练
 * @author xuwenfeng
 *
 */
@Slf4j
public class Yolo2Trainer {


    private static final String IMAGES_FOLDER="JPEGImages";
    
    private static final String ANNOTATIONS_FOLDER="Annotations";
    
    private static final String ANNOTATION_FORMAT=".xml";
   

    public static void main(String[] args) throws IOException {

         
    	String configFile="conf/Yolo2Hyperparameter-VOC2012-dectection.json";

      	File yolo2HyperparameterFile=new File(configFile);
    	
    	if(!yolo2HyperparameterFile.exists()) {
    		
    		log.error(configFile.concat(" does not exists !"));
    		
    		return;
    	}
    	
    	String yolo2HyperparameterFileJsonStr=FileUtils.readFileToString(yolo2HyperparameterFile, "UTF-8");
		
      	log.info(yolo2HyperparameterFileJsonStr);
    	
		Yolo2Hyperparameter yolo2Hyperparameter= JsonUtils.jsonToObject(yolo2HyperparameterFileJsonStr, Yolo2Hyperparameter.class);
		
        File imageDir = new File(yolo2Hyperparameter.getDataDir(), IMAGES_FOLDER);
        
        File annotationDir = new File(yolo2Hyperparameter.getDataDir(),ANNOTATIONS_FOLDER);
        //删除无用的系统文件
        deleteUselessFile(annotationDir);
        //删除无用的系统文件
        deleteUselessFile(imageDir);
        
        log.info("Load data...");
        
        Random random = new Random(yolo2Hyperparameter.getRandomSeed());
        //创建输入分割器数组
        InputSplit[] inputSplit = getInputSplit(imageDir, random,yolo2Hyperparameter);
        //训练集文件分割器
        InputSplit trainDataInputSplit = inputSplit[0];
        //测试集文件分割器
        InputSplit testDataInputSplit  = inputSplit[1];
        //创建训练记录读取数据集迭代器
        DataSetIterator trainRecordReaderDataSetIterator = getDataSetIterator(yolo2Hyperparameter,trainDataInputSplit);
        
        //DataSetIterator testRecordReaderDataSetIterator = getDataSetIterator(yolo2Hyperparameter,testDataInputSplit);
        
       //加载已有模型，如果本地不存在，则会从远程将预训练模型下载到当前用户的 
        //.deeplearning4j/models/tiny-yolo-voc_dl4j_inference.v2.zip 目录 
        ComputationGraph pretrainedComputationGraph =null;
        
        File latestModelFile=getLatestModelFile(yolo2Hyperparameter);
        
        if(latestModelFile==null) {
        	 pretrainedComputationGraph = (ComputationGraph) YOLO2.builder().build().initPretrained();
        }else {
             pretrainedComputationGraph = ModelSerializer.restoreComputationGraph(latestModelFile,true);
        }

       
        ComputationGraph model=null;
        
        if(latestModelFile==null) {
        	model=getTransferLearningModel(yolo2Hyperparameter,pretrainedComputationGraph);
        }else {
        	model=pretrainedComputationGraph;
        }
        log.info("\n Model Summary \n" + model.summary());

        log.info("Train model...");
        
        //设置监听器，每次迭代打印一次得分
        model.setListeners(new ScoreIterationListener(1));
        
        int startEpoch=0;
        
        if(latestModelFile!=null) {
        	startEpoch=getLatestModelFileIndex(latestModelFile);
        }
      
        long startTime=System.currentTimeMillis();
        
        String modelSavePath=yolo2Hyperparameter.getModelSavePath();
        
        if(!modelSavePath.endsWith(File.separator)) {
        	modelSavePath=modelSavePath.concat(File.separator);
        }
        
        for (int i = startEpoch; i < yolo2Hyperparameter.getEpochs(); i++) {
        	//每轮训练开始之前将数据集重置
            trainRecordReaderDataSetIterator.reset();
            
            model.fit(trainRecordReaderDataSetIterator);
      

            log.info("*** Completed epoch {} ***", i);
        }

        long endTime=System.currentTimeMillis();
        
        log.info("*** Completed all epoches at {} mins", (endTime-startTime)/(1000*60));
       
    }

	public static InputSplit[] getInputSplit(File imageDir, Random random,Yolo2Hyperparameter yolo2Hyperparameter) {
		
		 //随机路径过滤器，可以写规则来过滤掉不需要的数据
        RandomPathFilter pathFilter = new RandomPathFilter(random) {
            @Override
            protected boolean accept(String name) {
            	//转换为标签文件的路径
                name = name.replace(File.separator+IMAGES_FOLDER+File.separator, 
                		File.separator+ANNOTATIONS_FOLDER+File.separator)
                		.replace(yolo2Hyperparameter.getImageFormat(), ANNOTATION_FORMAT);
                log.info("loading annotation:"+name);
                try {
                	//如果图片文件对应的标签文件存在，则表示此条数据可以使用
                    return new File(new URI(name)).exists();
                } catch (URISyntaxException ex) {
                    throw new RuntimeException(ex);
                }
            }
        };
		InputSplit[] inputSplit = new FileSplit(
        		 imageDir,
        		 //允许的图片格式
        		 NativeImageLoader.ALLOWED_FORMATS,random)
        		 //按9:1的比例分割数据为训练集与测试集
        		.sample(pathFilter, 0.9, 0.1);
		return inputSplit;
	}
	
	public static DataSetIterator getDataSetIterator(Yolo2Hyperparameter yolo2Hyperparameter,InputSplit inputSplit) throws IOException {
		
		
		 //创建训练目标检测记录读取器
        ObjectDetectionRecordReader objectDetectionRecordReader = new ObjectDetectionRecordReader(
        		yolo2Hyperparameter.getInputHeight(),
        		yolo2Hyperparameter.getInputWidth(), 
        		yolo2Hyperparameter.getChannels(),
        		yolo2Hyperparameter.getGridHeight(), 
        		yolo2Hyperparameter.getGridWidth(), 
        		new VocLabelProvider(yolo2Hyperparameter.getDataDir()));
        
        //创建记录读取器监听器，可以在加载数据时进行相应处理
       // RecordListener recordListener=new ObjectDetectRecordListener();
        //设置读取器监听器
        //objectDetectionRecordReader.setListeners(recordListener);
        //初始化训练目标检测记录读取器
        objectDetectionRecordReader.initialize(inputSplit);
        //标签开始索引
        int labelIndexFrom=1;
        //标签结束索引
        int labelIndexTo=1;
        //是否为回归任务
        boolean regression=true;

        //创建训练记录读取数据集迭代器
        DataSetIterator recordReaderDataSetIterator = new RecordReaderDataSetIterator(
        		objectDetectionRecordReader, 
        		yolo2Hyperparameter.getBatchSize(),
        		labelIndexFrom,
        		labelIndexTo, 
        		regression);
        //设置图片预处理器，将像素值归一化到0-1之间
        recordReaderDataSetIterator.setPreProcessor(new ImagePreProcessingScaler(0, 1));
        
        return recordReaderDataSetIterator;
   	}

	public static ComputationGraph getTransferLearningModel(Yolo2Hyperparameter yolo2Hyperparameter,ComputationGraph pretrainedComputationGraph) {
		//先验边界框，可以用kmeans算法将所有人工标注的边界框进行聚类得到
        INDArray boundingBoxPriors= Nd4j.create(yolo2Hyperparameter.getBoundingBoxPriors());
        
        //调优配置
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
        		//使用CUDNN训练时设置此项
        		//.cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                .seed(yolo2Hyperparameter.getRandomSeed())
                //优化算法
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                //在梯度传递给updater之前进行梯度归一化
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                //梯度归一化阈值，仅在梯度归一化设置为GradientNormalization.ClipL2PerLayer起作用
                .gradientNormalizationThreshold(1.0)
                
                .l2(0.00001)
                //设置更新器
               
                .updater(new Adam.Builder().learningRate(yolo2Hyperparameter.getLearningRate()).build())
                //激活函数，因为YOLO是回归任务，所以用Activation.IDENTITY
                .activation(Activation.IDENTITY)
                //分数和梯度是否应除以小批量大小。大多数用户应该将此值保留为默认值true。
                .miniBatch(true)
                //开启工空间模式以结节内存
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .build();
       
       
        //迁移学习，并调优模型
        ComputationGraph model = new TransferLearning
        		 .GraphBuilder(pretrainedComputationGraph)
        		 //微调配置
                .fineTuneConfiguration(fineTuneConf)
                //设置输入类型
                .setInputTypes(InputType.convolutional(
                		yolo2Hyperparameter.getInputHeight(), 
                		yolo2Hyperparameter.getInputWidth(), 
                		yolo2Hyperparameter.getChannels()))
                //从计算图中删除指定的顶点，但保留其连接。
                //注意，这里的期望是，然后用相同的名称重新添加另一个顶点，
                //否则图形将保持无效状态，可能会引用不再存在的顶点。
                .removeVertexKeepConnections("conv2d_23")
                .removeVertexKeepConnections("outputs")
                .addLayer("conv2d_23",
                        new ConvolutionLayer.Builder(1, 1)
                                .nIn(1024)
                                .nOut(yolo2Hyperparameter.getBoxesNumber() * (5 + yolo2Hyperparameter.getClassesNumber()))
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Same)
                                .weightInit(WeightInit.UNIFORM)
                                .hasBias(false)
                                .activation(Activation.IDENTITY)
                                .build(), "leaky_re_lu_22")
                .addLayer("outputs",
                        new Yolo2OutputLayer.Builder()
                                .lambdaNoObj(yolo2Hyperparameter.getLamdbaNoObject())
                                .lambdaCoord(yolo2Hyperparameter.getLamdbaCoord())
                                .boundingBoxPriors(boundingBoxPriors)
                                .build(), "conv2d_23")
                .setOutputs("outputs")
                .build();
		return model;
	}

	private static void deleteUselessFile(File file) throws IOException {
		
		if(!file.exists()) {
			log.info("file not exists");
			return ;
		}
		
		File[] uselessFiles=file.listFiles();
        
        String uselessFilePrefix="._";
        
        for(File uselessFile:uselessFiles) {
        	if(uselessFile.getName().startsWith(uselessFilePrefix)) {
        		log.info("deleting ..."+uselessFile.getName());
        		FileUtils.deleteQuietly(uselessFile);
        	}
        }
	}
	
	/**
	 * 获取最新的模型
	 * @param yolo2Hyperparameter
	 * @return
	 */
	public static File getLatestModelFile(Yolo2Hyperparameter yolo2Hyperparameter) {
		
		File modelSavePath=new  File(yolo2Hyperparameter.getModelSavePath());
		
		File[] files=modelSavePath.listFiles();
		
		if(files==null) {
			return null;
		}
		
		List<File> fileList=new ArrayList<File>();
	
		for(File file:files) {
			if(file.getName().contains(yolo2Hyperparameter.getName())) {
				fileList.add(file);
			}
		}
		if(fileList.size()==0) {
			return null;
		}
		
		Collections.sort(fileList, new Comparator<File>() {
            public int compare(File f1, File f2) {
                long diff = f1.lastModified() - f2.lastModified();
                if (diff > 0)
                    return 1;
                else if (diff == 0)
                    return 0;
                else
                	//如果 if 中修改为 返回-1 同时此处修改为返回 1  排序就会是递减
                    return -1;
            }

            public boolean equals(Object obj) {
                return true;
            }

        });
		
		return fileList.get(0);
	}
	
	 public static int getLatestModelFileIndex(File file) {
		 
		 String[] fileNames=file.getName().split("_");
		 
		 String latestModelFileIndexStr=fileNames[fileNames.length-1];
		 
		 return Integer.parseInt(latestModelFileIndexStr);
	 }
	
	
}
