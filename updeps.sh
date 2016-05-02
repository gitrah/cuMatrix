#!/bin/sh
echo '
Updeps!
      s t a r t i n g   V M . . .
'
exec scala "$0" "$@"
!#

import scala.io._
import java.io._
import scala.collection.mutable.ListBuffer

// todo -e <path to executable or obj dir> to override/replace stamp file

object Dep {
  private val Tag = " lastModified "
  private val StampsFile = "-timestamps.txt"
  var Verbose = false
  var excludedSourcePathFragments = Set[String]()
  
  val exts = Array[String]("cc", "cu", "cpp", "c++", "c", "h", "hpp", "cuh")

  def sourceList(f: String): Set[File] = sourceList(new File(f))
  def sourceList(f: File): Set[File] = {
    println("	Scanning " + f + " for sources")
    if(!excludedSourcePathFragments.isEmpty) println( "excluded " + excludedSourcePathFragments)
      
    var s = Set[File]()
    // spelunk dirs
    s ++= (for (ff <- f.listFiles().filter(_.isDirectory).toList) yield (sourceList(ff))).flatten
    // filter dir children for sources
    s ++= (f.listFiles(new FilenameFilter() {
      def accept(d: File, name: String) = {
        !excludedQ(d.getAbsolutePath + File.separator + name) && exts.exists(ext => name.endsWith("." + ext))
      }
    })).toSet
    s
  }

  def readLastMods(timestamps: File) = {
    var m = Map[File, Long]()
    if (timestamps.exists())
      for (l <- Source.fromFile(timestamps).getLines()) {
        val sp = l.split(Tag)
        m += (new File(sp(0)) -> sp(1).toLong)
      }
    m
  }

  def lastMods(s: Set[File]) = {
    var m = Map[File, Long]()
    for (f <- s)
      m += (f -> f.lastModified())
    m
  }

  def saveLastMods(m: Map[File, Long], stampFilePath: String) {
    println(" Saving timestamps (" + stampFilePath +")...\n")
    val w = new FileWriter(stampFilePath)
    for (t <- m)
      w.write(t._1.getAbsolutePath + Tag + t._2 + "\n")
    w.flush
    w.close
  }
  
  def sameFileQ( a: File, b : File) = {
    //val r = 
    (a.getAbsolutePath().replace("/./", "/")).equals( (b.getAbsolutePath().replace("/./", "/"))) &&
     a.length == b.length && a.lastModified == b.lastModified
    //println(r +":\n\ta "+ a.getAbsolutePath().replace("/./", "/") + "\n\tb " +b.getAbsolutePath().replace("/./", "/")) 
    // r
  }
  
  def excludedQ( f : String) = excludedSourcePathFragments.exists( sx => f.contains(sx) )

  def dif(lm: Map[File, Long], om: Map[File, Long]) = {
    var fs = Set[File]()
    val ofs = om.keySet
    var printedGap = "\n"
    for (tup <- lm) {
      var samef : File = null
      for( of <- ofs; if samef == null) {
        if(sameFileQ(tup._1, of)) 
          samef = of
      }
      if(samef != null)
       om.get(samef) match {
        case Some(v) =>
          if (v > tup._2) {
            println(printedGap + tup._1 + " is older than targ by " + (v - tup._2) / 1000 + "s")
            printedGap = ""
          } else if (v < tup._2) {
            println(printedGap + tup._1 + " is newer than targ by " + (tup._2 - v) / 1000 + "s")
            fs += tup._1
            printedGap = ""
          }
        case _ =>
          println(printedGap + "new file " + tup._1)
          fs += tup._1
          printedGap = ""
      } else {
        println(printedGap + "no match for file " + tup._1)
        printedGap = ""
     }
    }
    if(printedGap == "") println
    fs
  }
  
  // want timestamp to be sibling of sources folder
  def timestampPath( sourcesPath : String ) = {
    val corrPath = if(sourcesPath.endsWith(File.separator)) sourcesPath.substring(0, sourcesPath.length-1) else sourcesPath 
    corrPath + StampsFile
  }

  def changedSources(path: String, save: Boolean) = {
    val allSources = sourceList(new File(path))
    println("	Checking timestamps")
    val lm = lastMods(allSources)
    val stampPath = timestampPath(path)
    val om = readLastMods(new File(stampPath))
    if (om.isEmpty || save) {
      saveLastMods(lm, stampPath)
    }
    (allSources, dif(lm, om))
  }

  def headerFiles(fs: Set[File]) = {
    fs.filter(_.getName.endsWith(".h"))
  }

  def importStatements(f: File) = {
    var s = Set[String]()
    for (l <- Source.fromFile(f).getLines) {
      if (l.startsWith("#include")) {
        val inc = l.substring("#include".length).trim
        s += (if (inc.indexOf("/") > -1) {
          l.substring(l.lastIndexOf("/") + 1, l.length - 1)
        } else {
          if (l.endsWith("\""))
            l.substring(l.indexOf("\"") + 1, l.length - 1)
          else
            l.substring(l.indexOf("<") + 1, l.length - 1)
        })
      }
    }
    s
  }

  def buildImportMap(allSources: Set[File]) = {
    println("	Building import map")
    var m = Map[File, Set[File]]()
    var mn = Map[String, File]()
    for (f <- allSources)
      mn += (f.getName() -> f)
    for (h <- allSources) {
      val imps = importStatements(h)
      if (!imps.isEmpty && !(mn.keySet.intersect(imps)).isEmpty)
        m += (h -> (for (i <- imps) yield {
          mn.get(i) match {
            case Some(fyl) => fyl
            case _         => Unit
          }
        }).filter(_ != Unit).asInstanceOf[Set[File]])
    }
    m
  }

  def headerQ(f: File) = f.getName.endsWith(".h")

  /*
     for each altered file (as determined from timestamp file)
    	for each source importing altered file
   			add to dependencies
   			if source is also header, add to altered file list 
   */

  def depsieDo(allSources: Set[File], m: Map[File,Set[File]], alteredFiles: Set[File]) = {
    println("	Generating dependency set for above modified files...\n")
    var deps = Set[File]() ++ alteredFiles
    var headersToCheck = Set[File]() ++ alteredFiles
    var headersAlreadyChecked = Set[File]()
    while (!headersToCheck.isEmpty) {
      val currHeader = headersToCheck.head
      if(Verbose)println("currHeader " + currHeader)
      headersToCheck = headersToCheck drop 1
      if(Verbose)println( " headersToCheck " + headersToCheck);
      // if the current source has imports
      for (currSrc <- allSources) {
        m.get(currSrc) match {
          case Some(impers) =>
            if(Verbose)println("currSrc " + currSrc + " imports " + impers)
            if (impers.contains(currHeader)) {
              if(Verbose)println(currSrc + " imports " + currHeader)
              deps += currSrc
              if(Verbose)println("after adding " + currSrc + " deps now " + deps)
              if (headerQ(currSrc) && !headersAlreadyChecked.contains(currSrc)) {
                if(Verbose)println("adding " + currSrc + " to headersToCheck") 
                headersToCheck += currSrc
              }
            }
          case _ => if(Verbose)println(currSrc + " doesn't import any of this project's sources")
        }
      }
      if(Verbose)println("adding currHeader "+ currHeader + " to headersAlreadyChecked " + headersAlreadyChecked)
      headersAlreadyChecked += currHeader
    }
    deps
  }
  
  def rebuilds(path: String, save: Boolean, touch: Boolean) = {
    val (allSources, newsies) = changedSources(path, save)
    
    if(newsies.size > 0) {
      val importMap = buildImportMap(allSources)
      val deps = if(newsies.size > 0) depsieDo(allSources, importMap, newsies) else Set[File]()
      // touch any calculated deps if requested
      val sdeps = deps.toList.sorted
      if (touch)
        for (f <- deps.diff(newsies)) {
          val nameNoExt = f.getName().substring(0,  f.getName().lastIndexOf("."))
          val objf = new File("."+ File.separator + "release" +File.separator + nameNoExt + ".o" )
          if (objf.exists) {
            println("removing " + objf)
            objf.delete()
          } else {
            println("updating " + f);
            f.setLastModified(System.currentTimeMillis())
          }
        }
      else {
        for (d <- sdeps) println(d)
      }
      sdeps
    } else {
      println("                          (samole samole)")
      Set[File]()
    }
  }

  def usage() {
    println(
      """Usage:
         
        updeps.sh [OPTION]...
        
  Scan for sources and timestamps, determine import graph and/or 
    touch dependent """ + (for (ex <-exts.toList.sorted) yield("." + ex)).mkString("{ ", ", ", " }") + """ sources.

  -r, -root (root source directory)
    If no directory is specified default folder './src' is used
    
  -x, -exclude <comma separated source path fragments> to exclude  

  -s, -save (save timestamps) if true (or unspecified) a file
    named <<SOURCE_FOLDER>>-timestamps.txt is created/overwritten 
    with timestamps of all discovered sources.
  
  -t, -touch (touch dependents) if true (or unspecified) 
    each file deteremined to be dependent upon any file that is 
    newer than its entry in the timestamp file is touched
    
  -v, -verbose if true show extra logging
  
""")
  }

  def optionQ(l : Set[String], keys : List[String], defaultValue : Boolean) ={
    var ret = defaultValue
    var remove = ""
    for( a <- l) 
       for( k <- keys) {
        val alc = a.toLowerCase
        val keyExp = "-"+k
        val keyEqExp = keyExp+"="
        val kidx = alc.indexOf(keyExp) 
        val keidx = alc.indexOf(keyEqExp) 
        if(kidx > -1) {
           val v = if(keidx > -1 ) alc.substring(keidx + keyEqExp.length) else ""
           try {
             ret= alc.equals(keyExp) || "t".equalsIgnoreCase(v) || "1".equalsIgnoreCase(v) ||  v.toBoolean
           } catch {
             case ex : IllegalArgumentException => {
               println( "unrecognized value " + v )
               throw ex
             }
               
           }
           remove = a
           if(Verbose) println("option " + k + " set to " + ret)
        }
     }
    (l - remove, ret)
  }
  
  def optionS(l : Set[String], keys : List[String], defaultValue : String)={
    var ret = defaultValue
    var remove = ""
    for( a <- l) 
      for( k <- keys) {
        val keyExp = "-"+k+"="
        val idx = a.toLowerCase.indexOf(keyExp) 
        if(idx > -1) {
           ret= a.substring(idx + keyExp.length)
          remove = a
          if(Verbose) println("option " + k + " set to " + ret)
        }
     }
    (l - remove, ret)
  }
  
  def main(args: Array[String]) {
    var argsS = args.toSet
    if (argsS.contains("--help")
      || argsS.contains("-help")
      || argsS.contains("-?")
      || argsS.contains("--?")) {
      usage()
    } else {
      try {
        var tup =  optionQ(argsS, List("v","verbose"), false)
        if(Verbose)println("after verbose tup._1 " + tup._1)
        Verbose = tup._2
        var tups = optionS(tup._1, List("r","root"), "./src")
        val path = tups._2
        if(Verbose)println("after path tups._1 " + tups._1)
        tup  = optionQ(tups._1, List("s","save"), false)
        val save = tup._2
        if(Verbose)println("after save tup._1 " + tup._1)
        tup  = optionQ(tup._1, List("t","touch"), false)
        val touch = tup._2
        if(Verbose)println("after touch tup._1 " + tup._1)
        tups =  optionS(tup._1, List("x","exclude"), "")
        val exSrcs = tups._2
        excludedSourcePathFragments = 
          if(!exSrcs.isEmpty) 
            exSrcs.split(",").toSet
         else Set[String]()
      
        if(!tups._1.isEmpty) {
          println("Unkown arguments : " + tups._1)
          usage()
        } else {
          rebuilds(path, save, touch)
        }
      } catch {
        case ex : Throwable => 
          println("got exception " + ex)
          usage()
      }
    }
    println("\ndone!\n")
  }
  
}
Dep.main(args)
