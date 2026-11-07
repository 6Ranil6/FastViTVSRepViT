package ru.bmstu.flowers;

import android.app.Activity;
import android.os.Bundle;
import android.os.Build;
import android.os.Debug;
import android.os.SystemClock;
import android.os.PowerManager;
import android.content.Intent;
import android.graphics.Bitmap;
import android.view.View;
import android.view.WindowManager;
import android.widget.*;
import org.json.*;
import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.*;
import java.util.concurrent.*;

/** Локальный эксперимент с фиксированной тестовой подвыборкой и заданной подготовкой RGB. Модели загружаются по одной. */
public final class MainActivity extends Activity {
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private volatile boolean cancelled = false;
    private JSONObject manifest;
    private JSONArray images;
    private TextView output;
    private ImageView preview;
    private Button start, save, next, stop;
    private CheckBox reverse;
    private Spinner threads;
    private int imageIndex = 0;
    private volatile String csv = "";
    private static final int WARMUP = 30, RUNS = 200, SAVE = 7;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout root = new LinearLayout(this); root.setOrientation(1);
        root.setPadding(24, 60, 24, 36);
        TextView title = new TextView(this); title.setText("FastViT-SA12 / RepViT-M0.9 · Flowers 102"); title.setTextSize(22); root.addView(title);
        TextView info = new TextView(this); info.setText("ExecuTorch 1.0 · XNNPACK · CPU FP32\nПотоки CPU:"); root.addView(info);
        threads = new Spinner(this); threads.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, new String[]{"1", "2", "4"})); root.addView(threads);
        reverse = new CheckBox(this); reverse.setText("Сначала RepViT (чередуйте порядок запусков)"); root.addView(reverse);
        preview = new ImageView(this); preview.setAdjustViewBounds(true); root.addView(preview, new LinearLayout.LayoutParams(-1, 320));
        next = button(root, "Следующее тестовое фото", v -> { imageIndex = (imageIndex + 1) % images.length(); showImage(); });
        start = button(root, "Сравнить модели", v -> runBenchmark());
        stop = button(root, "Остановить", v -> cancelled = true); stop.setEnabled(false);
        save = button(root, "Сохранить CSV", v -> {
            Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT).setType("text/csv").addCategory(Intent.CATEGORY_OPENABLE);
            intent.putExtra(Intent.EXTRA_TITLE, "flowers-" + System.currentTimeMillis() + ".csv"); startActivityForResult(intent, SAVE);
        }); save.setEnabled(false);
        output = new TextView(this); output.setTextIsSelectable(true); output.setTextSize(14);
        ScrollView scroll = new ScrollView(this); scroll.addView(output); root.addView(scroll, new LinearLayout.LayoutParams(-1, 0, 1));
        setContentView(root);
        try {
            manifest = new JSONObject(new String(readAsset("manifest.json"), StandardCharsets.UTF_8));
            images = manifest.getJSONArray("images");
            if (images.length() == 0) throw new IOException("Пустой набор изображений");
            output.setText("Обученные веса.\n" + "Тестовая подвыборка: " + images.length() + " фото. Полная test accuracy измеряется на Mac.\n30 прогревов + 200 замеров; подготовка изображения исключена из времени forward.\n");
            showImage();
        } catch (Exception e) {
            output.setText("Нет готового bundle. На Mac выполните coursework.bundle, затем пересоберите приложение.\n" + e);
            start.setEnabled(false); next.setEnabled(false);
        }
    }

    private Button button(LinearLayout parent, String label, View.OnClickListener listener) {
        Button b = new Button(this); b.setText(label); b.setOnClickListener(listener); parent.addView(b); return b;
    }
    private byte[] readAsset(String name) throws IOException {
        try (InputStream in = getAssets().open("bundle/" + name); ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buf = new byte[8192]; int n; while ((n = in.read(buf)) != -1) out.write(buf, 0, n); return out.toByteArray();
        }
    }
    private int size() throws JSONException { return manifest.getJSONObject("config").getInt("image_size"); }
    private void showImage() {
        try {
            JSONObject im = images.getJSONObject(imageIndex); byte[] rgb = readAsset(im.getString("file")); int s = size(); int[] colors = new int[s*s];
            for (int i=0; i<colors.length; i++) colors[i] = 0xff000000 | ((rgb[3*i]&255)<<16) | ((rgb[3*i+1]&255)<<8) | (rgb[3*i+2]&255);
            preview.setImageBitmap(Bitmap.createBitmap(colors, s, s, Bitmap.Config.ARGB_8888));
            next.setText("Фото " + im.getInt("id") + " · класс " + (im.getInt("target")+1) + " → следующее");
        } catch (Exception e) { append(e.toString()); }
    }
    private Tensor input(int index) throws Exception {
        byte[] rgb = readAsset(images.getJSONObject(index).getString("file")); int s=size(), area=s*s;
        if (rgb.length!=area*3) throw new IOException("Неверный размер RGB");
        JSONObject cfg=manifest.getJSONObject("config"); JSONArray mean=cfg.getJSONArray("mean"), std=cfg.getJSONArray("std");
        float[] x=new float[area*3];
        for(int c=0;c<3;c++) for(int i=0;i<area;i++) x[c*area+i]=((rgb[3*i+c]&255)/255f-(float)mean.getDouble(c))/(float)std.getDouble(c);
        return Tensor.fromBlob(x,new long[]{1,3,s,s});
    }
    private File modelFile(String key) throws Exception {
        File file=new File(getFilesDir(),key+".pte"); MessageDigest digest=MessageDigest.getInstance("SHA-256");
        try(InputStream in=getAssets().open("bundle/"+key+".pte"); FileOutputStream out=new FileOutputStream(file)) {
            byte[] buffer=new byte[8192]; int n; while((n=in.read(buffer))!=-1){digest.update(buffer,0,n);out.write(buffer,0,n);}
        }
        StringBuilder hex=new StringBuilder();
        for(byte b:digest.digest()) hex.append(String.format(Locale.ROOT,"%02x",b&255));
        String expected=manifest.getJSONObject("models").getJSONObject(key).getString("pte_sha256");
        if(!hex.toString().equals(expected)) throw new IOException("Контрольная сумма модели не совпадает");
        return file;
    }
    private static int argmax(float[] x) { int best=0; for(int i=1;i<x.length;i++) if(x[i]>x[best])best=i; return best; }
    private static double percentile(double[] values,double q) { double[] a=values.clone(); Arrays.sort(a); return a[(int)Math.ceil(q*a.length)-1]; }
    private int thermal() { return Build.VERSION.SDK_INT >= 29 ? ((PowerManager)getSystemService(POWER_SERVICE)).getCurrentThermalStatus() : -1; }
    private void checkCancelled() throws InterruptedException { if(cancelled || Thread.currentThread().isInterrupted()) throw new InterruptedException("Остановлено пользователем"); }
    private void append(String text) { runOnUiThread(() -> output.append(text+"\n")); }
    private void busy(boolean value) {
        start.setEnabled(!value); next.setEnabled(!value); threads.setEnabled(!value); reverse.setEnabled(!value); stop.setEnabled(value); save.setEnabled(!value && !csv.isEmpty());
        if(value)getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON); else getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }
    private void runBenchmark() {
        cancelled=false; csv=""; busy(true);
        int nthreads=Integer.parseInt(threads.getSelectedItem().toString());
        String[] order=reverse.isChecked()?new String[]{"repvit","fastvit"}:new String[]{"fastvit","repvit"};
        final int sample=imageIndex;
        worker.execute(() -> {
            StringBuilder rows=new StringBuilder("record,model,index,value,unit,target,prediction,reference_prediction\n");
            try {
                rows.append("meta,device,0,\"").append(Build.MANUFACTURER).append(' ').append(Build.MODEL).append("\",text,,,\n");
                rows.append("meta,android,0,").append(Build.VERSION.SDK_INT).append(",api,,,\n");
                rows.append("meta,threads,0,").append(nthreads).append(",count,,,\n");
                rows.append("meta,time,0,").append(System.currentTimeMillis()).append(",unix_ms,,,\n");
                rows.append("meta,benchmark_image_id,0,").append(images.getJSONObject(sample).getInt("id")).append(",id,,,\n");
                rows.append("meta,warmup,0,").append(WARMUP).append(",count,,,\n");
                rows.append("meta,measurements,0,").append(RUNS).append(",count,,,\n");
                rows.append("meta,backend,0,XNNPACK_CPU_FP32,text,,,\n");
                Tensor bench=input(sample);
                for(String key:order) {
                    String variant=manifest.getJSONObject("models").getJSONObject(key).getJSONObject("config").getJSONObject("models").getString(key);
                    rows.append("meta,").append(key).append("_variant,0,").append(variant).append(",text,,,\n");
                    checkCancelled(); append("Загрузка "+variant+"…");
                    File file=modelFile(key); long pssBefore=Debug.getPss(); long loadStart=SystemClock.elapsedRealtimeNanos();
                    Module module=Module.load(file.getAbsolutePath(),Module.LOAD_MODE_FILE,nthreads);
                    try {
                        double loadMs=(SystemClock.elapsedRealtimeNanos()-loadStart)/1e6;
                        EValue value=EValue.from(bench);
                        for(int i=0;i<WARMUP;i++){checkCancelled();module.forward(value);}
                        float[] demo=module.forward(value)[0].toTensor().getDataAsFloatArray();
                        int demoClass=argmax(demo); double sumExp=0.;
                        for(float logit:demo)sumExp+=Math.exp(logit-demo[demoClass]);
                        append(String.format(Locale.ROOT,"Фото %d → класс %d (softmax %.1f%%), правильный класс %d",images.getJSONObject(sample).getInt("id"),demoClass+1,100./sumExp,images.getJSONObject(sample).getInt("target")+1));
                        long pssLoaded=Debug.getPss(); int thermalBefore=thermal();
                        double[] times=new double[RUNS];
                        for(int i=0;i<RUNS;i++) {
                            checkCancelled(); long begin=SystemClock.elapsedRealtimeNanos(); module.forward(value);
                            times[i]=(SystemClock.elapsedRealtimeNanos()-begin)/1e6;
                            rows.append("latency,").append(key).append(',').append(i).append(',').append(times[i]).append(",ms,,,\n");
                        }
                        int thermalAfter=thermal(); long pssAfter=Debug.getPss();
                        int correct=0,agree=0,processed=0; double maxError=0.; int[][] cm=new int[102][102];
                        for(int i=0;i<images.length();i++) {
                            checkCancelled(); JSONObject image=images.getJSONObject(i);
                            float[] logits=module.forward(EValue.from(input(i)))[0].toTensor().getDataAsFloatArray();
                            if(logits.length!=102)throw new IOException("Ожидались 102 выхода");
                            int predicted=argmax(logits),target=image.getInt("target");
                            JSONArray ref=image.getJSONObject("reference_logits").getJSONArray(key); float[] refArray=new float[102];
                            for(int j=0;j<102;j++){refArray[j]=(float)ref.getDouble(j);maxError=Math.max(maxError,Math.abs(logits[j]-refArray[j]));}
                            int refPred=argmax(refArray); if(predicted==target)correct++;if(predicted==refPred)agree++;cm[target][predicted]++;processed++;
                            rows.append("prediction,").append(key).append(',').append(image.getInt("id")).append(",1,count,").append(target).append(',').append(predicted).append(',').append(refPred).append('\n');
                        }
                        double mean=Arrays.stream(times).average().orElse(0); double f1=0;
                        for(int c=0;c<102;c++){int actual=0,predicted=0;for(int j=0;j<102;j++){actual+=cm[c][j];predicted+=cm[j][c];}if(actual+predicted>0)f1+=2.0*cm[c][c]/(actual+predicted);}
                        double[] values={mean,percentile(times,.5),percentile(times,.95),correct/(double)processed,f1/102,agree/(double)processed,maxError,loadMs,file.length(),pssBefore,pssLoaded,pssAfter,thermalBefore,thermalAfter};
                        String[] names={"mean_ms","p50_ms","p95_ms","subset_accuracy","subset_macro_f1","reference_agreement","max_abs_logit_error","load_ms","model_bytes","pss_before_kib","pss_loaded_kib","pss_after_kib","thermal_before","thermal_after"};
                        for(int i=0;i<values.length;i++)rows.append("summary,").append(key).append(',').append(names[i]).append(',').append(values[i]).append(",value,,,\n");
                        rows.append("meta,").append(key).append(",sha256,").append(manifest.getJSONObject("models").getJSONObject(key).getString("pte_sha256")).append(",text,,,\n");
                        append(String.format(Locale.ROOT,"%s: p50 %.2f ms · p95 %.2f ms\nПодвыборка: %d/%d (%.1f%%) · совпадение с Mac: %d/%d\nPSS после прогрева: %d КиБ; max |Δlogit| %.5f",key,percentile(times,.5),percentile(times,.95),correct,processed,100.*correct/processed,agree,processed,pssLoaded,maxError));
                        if(agree!=processed || maxError>0.02)append("ВНИМАНИЕ: расхождение с эталоном; проверьте экспорт до выводов о качестве.");
                        if(thermalAfter>thermalBefore || thermalAfter>=2)append("Устройство нагревается: дайте остыть и повторите измерение.");
                    } finally {module.destroy();}
                }
                csv=rows.toString();
                try(FileOutputStream out=new FileOutputStream(new File(getFilesDir(),"last_results.csv"))){out.write(csv.getBytes(StandardCharsets.UTF_8));}
                append("Готово. Сохраните CSV. PSS — память всего процесса в отдельных точках, не пик памяти модели.");
            } catch(Exception e) { append("Прогон не завершён: "+e); }
            finally {runOnUiThread(() -> busy(false));}
        });
    }
    @Override protected void onActivityResult(int request,int result,Intent data) {
        super.onActivityResult(request,result,data);
        if(request==SAVE && result==RESULT_OK && data!=null && data.getData()!=null) {
            try(OutputStream out=getContentResolver().openOutputStream(data.getData())) {
                if(out==null)throw new IOException("Не удалось открыть файл");out.write(csv.getBytes(StandardCharsets.UTF_8));append("CSV сохранён.");
            } catch(Exception e){append(e.toString());}
        }
    }
    @Override protected void onDestroy(){cancelled=true;worker.shutdownNow();super.onDestroy();}
}
