package cn.skymind.examples.objectdetection;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Yolo2Hyperparameter {
	
	    private  String name;
	     
	    private  int classesNumber ;
	    
	    private  double[][] boundingBoxPriors;
	    
        private  double learningRate;
	    
	    private  String dataDir;
	    
	    private  int batchSize ;
	    
	    private  int epochs ;
	    
	    private  int inputWidth ;
	    
	    private  int inputHeight ;
	    
	    private  int channels;
	    
	    private  int gridWidth ;
	    
	    private  int gridHeight;
	    
	    private  int boxesNumber;
	   
	    private  int randomSeed ;

	    private  double lamdbaCoord;
	    
	    private  double lamdbaNoObject;
	    
	    private  String  imageFormat;
	    
	    private  String  modelSavePath;
	
	

}
