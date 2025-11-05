'''
author: Pufferfish404
date: 2025-06-16
last modified: 2025-07-10
description: 
    QuPath Script to generate tissue segmentation
'''

import static qupath.lib.gui.scripting.QPEx.*
import qupath.opencv.ml.pixel.*
import qupath.lib.images.writers.ImageWriterTools
import qupath.lib.objects.classes.*
import ij.plugin.frame.RoiManager
import qupath.lib.images.servers.LabeledImageServer
import qupath.lib.regions.*
import ij.*
import java.awt.Color
import java.awt.image.BufferedImage
import qupath.lib.images.servers.ImageServerProvider
import java.util.Scanner.*
import qupath.lib.images.servers.*
import qupath.lib.images.servers.bioformats.BioFormatsServerBuilder //script will work best for TIFF
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import qupath.lib.gui.QuPathGUI
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.*
import java.util.ArrayList;
import javax.imageio.ImageIO;
import qupath.lib.images.ImageData;
import qupath.lib.io.PathIO;
import qupath.lib.io.PathIO.*;

//def project = getProject()
//def argsList = getArgs()
def inputDir = new File(args[0])
def outputDir = new File(args[1])

if (!outputDir.exists()) {
    mkdirs(outputDir.toString())
}
def imageFiles = inputDir.listFiles({ f -> f.name.endsWith("DAPI.tiff") } as FileFilter)

//for (entry in project.getImageList()) {
for (file in imageFiles) {
    //set up thresholder
    boolean criticalRegion = false
    boolean normProc = true
    int thresholdVal = 250
    int thresholdIncrement = 25
    int thresholdAdjust = 0
    float differential = 1.00
    float criticalDiff = .909
    double prevArea = 0
    double currentArea = 0
    double area = 0
    double initialArea = 0
    def thresh = new File(args[2])
    def threshContents = thresh.getText('UTF-8')
    def modContents = threshContents.replaceAll("\"thresholds\": \\[\n\\s*\\d*.0",
                "\"thresholds\": \\[\n\t\s\s\s\s${thresholdVal}\\.0")
    thresh.write(modContents, 'UTF-8')
    
    //println(thresh.getText('UTF-8'))

    //set up image
    def imageName = file.getName() //entry.getImageName()
    //def imagePath = ("//prfs.hhmi.org/lilab/2025-06-11 - test 1% staining quality/${imageName}").toString()
    //def imageFile = new File(imagePath)
    def imagePathString = file.getAbsolutePath() //imageFile.getAbsolutePath()
    def server = ImageServerProvider.buildServer(imagePathString, BufferedImage.class, new String[0])
    def imageData = new ImageData(server)
    imageData.setImageType(ImageData.ImageType.FLUORESCENCE)
    def hierarchy = imageData?.getHierarchy()

    //println(imageName)
    
    //make sure no extraneous annotations are present
    clearAllObjects();
    
    //store prevArea once
    def thresPath = Paths.get(thresh.getAbsolutePath())
    def thresholder = new PixelClassifiers().readClassifier(thresPath)
    clearAllObjects();
    deselectAll(hierarchy);
    PixelClassifierTools.createAnnotationsFromPixelClassifier(imageData, thresholder, (double)2.5E12, (double)2.5E10, PixelClassifierTools.CreateObjectOptions.DELETE_EXISTING)
    selectAnnotations(hierarchy);
    PixelClassifierTools.addMeasurementsToSelectedObjects(imageData, thresholder, "")
    def annotations = hierarchy.getAnnotationObjects()
    for (ann in annotations) {
        if (ann.getMeasurementList().get("Positive area µm^2") < area || annotations.size() == 1) {
            area = ann.getMeasurementList().get("Positive area µm^2")
            }
    }
    clearAllObjects();
    prevArea = area
    initialArea = area
    //iteratively create a mask until conditions met
    while ((differential > criticalDiff && criticalRegion == false) || (differential < criticalDiff && criticalRegion == true)) {
        //println(prevArea)
        thresholdVal += thresholdIncrement
        threshContents = thresh.getText('UTF-8')
        modContents = threshContents.replaceAll("\"thresholds\": \\[\n\\s*\\d*.0",
                    "\"thresholds\": \\[\n\t\s\s\s\s${thresholdVal}\\.0")
        thresh.write(modContents, 'UTF-8')
        thresholder = new PixelClassifiers().readClassifier(thresPath)
        //for (i : hierarchy.getAnnotationObjects().size()) {
            //println(hierarchy.getAnnotationObjects()[i - 1])
            //}
        clearAllObjects();
        deselectAll(hierarchy);
        PixelClassifierTools.createAnnotationsFromPixelClassifier(imageData, thresholder, (double)2.5E12, (double)2.5E10, PixelClassifierTools.CreateObjectOptions.DELETE_EXISTING)
        selectAnnotations(hierarchy);
        PixelClassifierTools.addMeasurementsToSelectedObjects(imageData, thresholder, "")
        annotations = hierarchy.getAnnotationObjects()
        for (ann in annotations) {
            if (ann.getMeasurementList().get("Positive area µm^2") < area) {
                area = ann.getMeasurementList().get("Positive area µm^2")
                }
        }
        currentArea = area
        differential = currentArea / prevArea
        if (differential < criticalDiff) {
            criticalRegion = true
            criticalDiff = 0.975
            thresholdIncrement = 10
            }
        if (currentArea < (.505 * initialArea)) {
            normProc = false
            thresholdAdjust = 50
            differential = 2.00
            criticalRegion = true
            }
        clearAllObjects();
        deselectAll(hierarchy);
        prevArea = currentArea
/**
        println("critical region is: " + criticalRegion)
        println("threshold set to: " + thresholdVal)
        println("currentArea:prevArea ratio equals: " + differential)
        println(area)
*/
    }
    
    //repeat creation step one more time for more fine shaving of edges 

    clearAllObjects();
    deselectAll(hierarchy);
    thresholdVal = thresholdVal + thresholdIncrement - thresholdAdjust
    threshContents = thresh.getText('UTF-8')
    modContents = threshContents.replaceAll("\"thresholds\": \\[\n\\s*\\d*.0",
                "\"thresholds\": \\[\n\t\s\s\s\s${thresholdVal}\\.0")
    thresh.write(modContents, 'UTF-8')
    thresholder = new PixelClassifiers().readClassifier(thresPath)
    PixelClassifierTools.createAnnotationsFromPixelClassifier(imageData, thresholder, (double)2.5E12, (double)2.5E10, PixelClassifierTools.CreateObjectOptions.DELETE_EXISTING)
    selectAnnotations(hierarchy);
    PixelClassifierTools.addMeasurementsToSelectedObjects(imageData, thresholder, "")
    
/**
    println("critical region is: " + criticalRegion)
    println("threshold set to: " + thresholdVal)
    println("currentArea:prevArea ratio equals: " + differential)
    println(annotations[0].getMeasurementList().get("Positive area µm^2"))
*/
    //filter through stored annotations for smallest mask & attach value of "Positive"
    annotations = hierarchy.getAnnotationObjects();
    //annotations[0].setPathClass(PathClass.fromString("Positive"))
    //println(annotations[0].getMeasurementList().get("Positive area µm^2")) //gets area

    //export GeoJSON
    def ROIPath = buildFilePath(outputDir.toString(), "ROIs_asGeoJSON")
    if (!Files.exists(Paths.get(ROIPath))) {
        mkdirs(ROIPath)
    }
    def ROIFile = new File(ROIPath, imageName.substring(0, imageName.length() - 5) + ".geojson")
    PathIO.exportObjectsAsGeoJSON(ROIFile, annotations, PathIO.GeoJsonExportOptions.FEATURE_COLLECTION)

    //create mask preview
    //server = ImageServerProvider.buildServer(imagePathString, BufferedImage.class, new String[0])
    def roi = annotations[0].getROI()
    double downsample = 72.0
    def request = RegionRequest.createInstance(server.getPath(), downsample, roi)
    def img = server.readRegion(request)
    def shape = roi.getShape()
    def imgMask = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_BYTE_GRAY)
    def g2d = imgMask.createGraphics()
    g2d.scale(1.0/request.getDownsample(), 1.0/request.getDownsample())
    g2d.translate(-request.getX(), -request.getY())
    g2d.setColor(Color.WHITE)
    g2d.fill(shape)
    g2d.dispose()
    def maskPath = buildFilePath(outputDir.toString(), "masks")
    if (!Files.exists(Paths.get(maskPath))) {
        mkdirs(maskPath)
    }
    def maskFile = new File(maskPath, "${imageName}_mask.png")
    ImageIO.write(imgMask, "PNG", maskFile)
    //new ImagePlus("Mask of ${imageName}", imgMask).show()s
}
println("job complete!")
