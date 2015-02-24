package org.opencv.samples.mathrecog;

import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by VladimirGerasimov on 22.02.2015.
 */
public class Container {
    private List<Mat> img = new ArrayList<Mat>();
    private List<String> st = new ArrayList<String>();

    Container() {
    }

    public void Add(Mat m, String s) {
        this.img.add(m);
        this.st.add(s);
    }

    public int count() {
        return this.st.size();
    }
}
