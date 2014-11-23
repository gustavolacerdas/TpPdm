package br.ufmg.pdm;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

import scala.Tuple2;
public class SVM {

	private static final String PATH_CENTROIDES = "/user/root/test/centroides.txt";
	private static final String PATH_MAP_BY_JOB_ID = "/user/root/test/recursosById.txt";
			
	public static void main(String[] args) {
	    SparkConf conf = new SparkConf().setAppName("SVM Classifier Example");
	    SparkContext sc = new SparkContext(conf);
	    
		//Centroides obtidos pelo KNN
	    JavaRDD<LabeledPoint> training = MLUtils.loadLibSVMFile(sc, PATH_CENTROIDES).toJavaRDD();;
	    training.cache();
	    
	    //Dados dos jobs
	    JavaRDD<LabeledPoint> test = MLUtils.loadLibSVMFile(sc, PATH_MAP_BY_JOB_ID).toJavaRDD();;
	    test.cache();
	    
	    // Run training algorithm to build the model.
	    int numIterations = 100;
	    final SVMModel model = SVMWithSGD.train(training.rdd(), numIterations);
	    
	    // Clear the default threshold.
	    model.clearThreshold();

	    // Compute raw scores on the test set.
	    JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test.map(
	      new Function<LabeledPoint, Tuple2<Object, Object>>() {
	        public Tuple2<Object, Object> call(LabeledPoint p) {
	          Double score = model.predict(p.features());
	          return new Tuple2<Object, Object>(score, p.label());
	        }
	      }
	    );
	    
	    // Get evaluation metrics.
	    BinaryClassificationMetrics metrics = 
	      new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
	    double auROC = metrics.areaUnderROC();
	    
	    System.out.println("Area under ROC = " + auROC);
	  }

}
