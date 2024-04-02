package com.example.validation;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.os.Environment;
import android.util.Log;

import com.example.validation.ml.AnnClassifier;
import com.example.validation.ml.AnnMulticlass;
import com.example.validation.ml.Apnea;
import com.example.validation.ml.ArrhythmiaOnEcgClassification;
import com.example.validation.ml.CnnMulticlass;
import com.example.validation.ml.HybridMulticlass;
import com.example.validation.ml.LstmMulticlassNew;
import com.example.validation.ml.RnnLstmMulticlass;
import com.example.validation.ml.RnnSimpleMulticlass;
import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Collectors;

public class MainActivity extends AppCompatActivity {
    long startTimeMillis  = System.currentTimeMillis();
    Date currentDate = new Date(startTimeMillis);
    SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss.SSS", Locale.getDefault());
    String formattedDateTime = sdf.format(currentDate);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        int k = 0;
        int[] predictClassCount = new int[2];
        String folderPath = Environment.getExternalStorageDirectory().getAbsolutePath() + "/Data ";
        CopyOnWriteArrayList<String[]> data = new CopyOnWriteArrayList<>();
        CSVWriter writer = null;
        String csvforECG = null;
        File externalDir = getApplicationContext().getExternalFilesDir(null);
        File folder = new File(externalDir, folderPath);
        if (!folder.exists()) {
            if (folder.mkdirs()) {
                // Folder created successfully
                csvforECG = new File(folder, "ECGData.csv").getPath();
            } else {
                // Failed to create folder
                System.out.println("Can't Create Folder");
            }
        }

        try (InputStream inputStream = getResources().openRawResource(R.raw.features_test);
             InputStreamReader inputStreamReader = new InputStreamReader(inputStream);
             BufferedReader reader = new BufferedReader(inputStreamReader);
             CSVReader csvReader = new CSVReader(reader)) {

            String[] line;
            while ((line = csvReader.readNext()) != null) {
                List<Double> justEcg = Arrays.stream(line)
                        .map(Double::parseDouble)
                        .collect(Collectors.toList());

                try {
                    File fileECG = new File(folder, "Label.csv");
                    if (!fileECG.exists()) {
                        fileECG.createNewFile();
                    }
                    writer = new CSVWriter(new FileWriter(fileECG, true));
                    try {
                        int predictClass = 0;
                        Apnea model = Apnea.newInstance(this);

                        // Creates inputs for reference.
                        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 40}, DataType.FLOAT32);


                        // Find the minimum and maximum values of the ECG data
                        double ecgMin = Double.MAX_VALUE;
                        double ecgMax = Double.MIN_VALUE;
                        for (Double sample : justEcg) {
                            if (sample < ecgMin) {
                                ecgMin = sample;
                            }
                            if (sample > ecgMax) {
                                ecgMax = sample;
                            }
                        }

                        // Pack normalized ECG data into a ByteBuffer
                        ByteBuffer byteBuffer = ByteBuffer.allocate(40 * 4);
                        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
                        for (Double sample : justEcg) {
                            float samplefloat = (float) ((sample - ecgMin) / (ecgMax - ecgMin));
                            if (byteBuffer.remaining() == 0)
                                break;
                            byteBuffer.putFloat(samplefloat);
                        }
                        byteBuffer.position(0);


                        inputFeature0.loadBuffer(byteBuffer);

                        // Runs model inference and gets result.
                        Apnea.Outputs outputs = model.process(inputFeature0);
                        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                        // Get predicted class
                        float[] scores = outputFeature0.getFloatArray();
                        for(float score : scores){
                            System.out.println(score);
                        }

                        predictClass = getMaxIndex(scores);
                        //data.add(new String[]{String.valueOf(k), String.valueOf(predictClass)});

                        predictClassCount[predictClass]++;
                        System.out.println("Predict Class : " + k + " " + predictClass);

                        // Releases model resources if no longer used.
                        model.close();
                        calculateAndLogElapsedTime();
                        k++;
                        writer.writeAll(Collections.singleton(new String[]{String.valueOf(k), String.valueOf(predictClass)}));
                    } catch (IOException e) {
                        // TODO Handle the exception
                    }
                    writer.close();
                }
                catch (IOException e){
                    e.printStackTrace();
                }
            }

//            for(int i=0;i<predictClassCount.length;i++){
//                System.out.println("All Classes : " + i + " " + predictClassCount[i]);
//            }


        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public static int getMaxIndex(float[] array) {
        int maxIndex = 0;
        float maxValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static int getMaxIndexforANN(float[] array) {
        int maxIndex = 0;
        for(int i=0;i<array.length;i++){
            if(array[i] > maxIndex)
                return 0;
        }
        return 1;
    }

    private void calculateAndLogElapsedTime() {
        long currentTimeMillis = System.currentTimeMillis();

        long elapsedTimeMillis = currentTimeMillis - startTimeMillis;

        Date elapsedTimeDate = new Date(elapsedTimeMillis);

        SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss.SSS", Locale.getDefault());

        String formattedElapsedTime = sdf.format(elapsedTimeDate);

        //Log.d("ElapsedTime", "Elapsed Time: " + formattedElapsedTime);
    }
}