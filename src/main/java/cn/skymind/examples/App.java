package cn.skymind.examples;

import cn.skymind.examples.classification.InceptionResNetV1Classification;
import cn.skymind.examples.classification.ResNet50Classification;
import cn.skymind.examples.classification.Vgg19Classification;
import cn.skymind.examples.objectdetection.Yolo2Trainer;
import cn.skymind.examples.utils.AppRunOptions;
import com.beust.jcommander.JCommander;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class App {

    @SuppressWarnings("static-access")
    public static void main( String[] args ){

        AppRunOptions appRunOptions=new AppRunOptions();

        JCommander jCommander = JCommander.newBuilder()
                .addObject(appRunOptions)
                .build();

        jCommander.parse(args);

        if (appRunOptions.isHelp()) {
            jCommander.usage();
            return;
        }

        log.info(appRunOptions.getEntryClass()+" start ");

            try {


                if(appRunOptions.getEntryClass().equals(InceptionResNetV1Classification.class.getSimpleName())){

                        InceptionResNetV1Classification.train();

                }else if(appRunOptions.getEntryClass().equals(ResNet50Classification.class.getSimpleName())) {

                        ResNet50Classification.train();

                }else if(appRunOptions.getEntryClass().equals(Vgg19Classification.class.getSimpleName())) {

                        Vgg19Classification.train();

                }else if(appRunOptions.getEntryClass().equals(Yolo2Trainer.class.getSimpleName())){

                        Yolo2Trainer.main(null);
                }


            } catch (Exception e) {

                log.error("",e);
            }

        log.info(appRunOptions.getEntryClass()+" end ");



    }
}
