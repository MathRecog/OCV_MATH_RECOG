package org.opencv.samples.mathrecog;

import android.os.Environment;
import android.util.Log;

import org.opencv.core.*;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.Core.*;
import static org.opencv.highgui.Highgui.*;
import static org.opencv.imgproc.Imgproc.*;

/**
 * Created by VladimirGerasimov on 05.02.2015.
 */
public class ImageProcessor {
    private Mat photo;
    String debug;
    ImageProcessor(String img) {
        photo = imread(img, CV_LOAD_IMAGE_COLOR);
        if(photo.empty()) {
            throw new Error("Cannot find the image file: " + img);
        }
        File f = new File(img);
        debug = f.getName();
    }

    public void Process(){
        Mat patient = photo;
        cvtColor(patient, patient, COLOR_BGR2GRAY, 0);
            imwrite("/sdcard/MathRecogData/__gs__" + debug, patient);
        adaptiveThreshold(patient, patient, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 15);
            imwrite("/sdcard/MathRecogData/__ats__" + debug, patient);
        Canny(patient, patient, 200.0, 295.0, 3, true);
            imwrite("/sdcard/MathRecogData/__canny__" + debug, patient);

        Mat hierarchy = new Mat();
        List<MatOfPoint> conts = new ArrayList<MatOfPoint>();
        findContours(patient, conts, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_TC89_KCOS);
        List<Mat> separatedConts = new ArrayList<Mat>();
        for( int i = 0; i < conts.size(); i++ ){
            Rect rect = boundingRect(conts.get(i));
            rectangle(patient, rect.tl(), rect.br(), new Scalar(255,255,255));
            Mat cropped = photo.submat(rect);
            separatedConts.add(cropped);
        }

        imwrite("/sdcard/MathRecogData/" + debug, patient);

    }
}
