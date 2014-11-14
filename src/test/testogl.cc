#include "../CuMatrix.h"
#include "../util.h"
#include "tests.h"
#include <GL/gl.h>
#include <GL/glut.h>
#include "../ogl/AccelEvents.h"
#include "../ogl/Point.h"
#include "../ogl/Perspective.h"
#include "../ogl/Arrow.h"
#include "../ogl/LookAt.h"
#include "../ogl/Text.h"

#ifdef 	CuMatrix_Enable_Ogl

void display(void)
{
/*  clear all pixels  */
    glClear (GL_COLOR_BUFFER_BIT);

    glColor3f (1.0, 1.0, 1.0);
    glBegin(GL_POINTS);
        glVertex3f (0.25, 0.25, 0.0);
        glVertex3f (0.75, 0.25, 0.0);
        glVertex3f (0.75, 0.75, 0.0);
        glVertex3f (0.25, 0.75, 0.0);
        glutSolidCube(2);
    glEnd();


    glFlush ();
}

void init (void)
{
/*  select clearing (background) color       */
    glClearColor (0.0, 0.0, 0.0, 0.0);

/*  initialize viewing values  */
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
}

void reshape(int x, int y) {
	printf("reshape! %d %d\n", x,y);
}

void keybd(unsigned char key, int x, int y) {
	printf("keybd %c %d %d\n", key, x,y);
}

void mouse(int button, int state, int x, int y) {
	printf("mouse btn %d state %d %d %d\n", button, state, x,y);
}

void idle() {
	printf("^");
}

template int testOglHelloworld<float>::operator()(int argc, char const ** args) const;
template int testOglHelloworld<double>::operator()(int argc, char const ** args) const;
template <typename T> int testOglHelloworld<T>::operator()(int argc, const char** args) const {
    glutInit(&argc, (char **)args);
    glutInitDisplayMode (GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize (250, 250);
    glutInitWindowPosition (100, 100);
    glutCreateWindow ("hello");
    init ();
    glutReshapeFunc(reshape);
    glutDisplayFunc(display);
    glutKeyboardFunc(keybd);
    glutMouseFunc(mouse);
   // glutIdleFunc(idle);
    glutMainLoop();

	return 0;
}

//
//
//
//
//

static GLfloat spin = 0.0;

void initAnim0(void)
{
   glClearColor (0.0, 0.0, 0.0, 0.0);
   glShadeModel (GL_FLAT);
}

void displayAnim0(void)
{
   glClear(GL_COLOR_BUFFER_BIT);
   glPushMatrix();
   glRotatef(spin, 0.0, 1.0, 1.0);
   glColor3f(1.0, 1.0, 1.0);
   glRectf(-25.0, -25.0, 25.0, 25.0);
   glutWireCube(10);
   glPopMatrix();
   glutSwapBuffers();
}

void spinDisplay(void)
{
   spin = spin + 2.0;
   if (spin > 360.0)
      spin = spin - 360.0;
   glutPostRedisplay();
}

void reshapeAnim0(int w, int h)
{
	printf("reshaped");
   glViewport (0, 0, (GLsizei) w, (GLsizei) h);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glOrtho(-50.0, 50.0, -50.0, 50.0, -300.0, 300.0);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}

void mouseAnim0(int button, int state, int x, int y)
{
   switch (button) {
      case GLUT_LEFT_BUTTON:
         if (state == GLUT_DOWN)
            glutIdleFunc(spinDisplay);
         break;
      case GLUT_MIDDLE_BUTTON:
         if (state == GLUT_DOWN)
            glutIdleFunc(NULL);
         break;
      default:
         break;
   }
}
template int testOglAnim0<float>::operator()(int argc, char const ** args) const;
template int testOglAnim0<double>::operator()(int argc, char const ** args) const;
template <typename T> int testOglAnim0<T>::operator()(int argc, const char** args) const {
	glutInit(&argc, (char**)args);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(250, 250);
	glutInitWindowPosition(100, 100);
	glutCreateWindow(args[0]);
	initAnim0();
	glutDisplayFunc(displayAnim0);
	glutReshapeFunc(reshapeAnim0);
	glutMouseFunc(mouseAnim0);
	glutMainLoop();
	return 0;
}

//
//
//  testOglPointAnim
//
//

void initPointAnim(void)
{
   glClearColor (0.0, 0.0, 0.0, 0.0);
   glShadeModel (GL_FLAT);
}

template <typename T> void mousePointAnim(int button, int state, int x, int y)
{
	outln("mousePointAnim here");
   switch (button) {
      case GLUT_LEFT_BUTTON:
         if (state == GLUT_DOWN)
            glutIdleFunc(Steppable<T>::stepCb);
         break;
      case GLUT_MIDDLE_BUTTON:
         if (state == GLUT_DOWN)
            glutIdleFunc(NULL);
         break;
      default:
         break;
   }
}
Perspective* thePersp;
LookAt* theLookAt;
double* p1;
double* p2;
double* p3;

double *thePoints[6];// = {&theLookAt->eyeX,&theLookAt->eyeY,&theLookAt->eyeZ,&theLookAt->centerX,&theLookAt->centerY,&theLookAt->centerZ};
const char* theVars[] = {"eyeX","eyeY","eyeZ","centerX","centerY","centerZ"};
int idx = 0;
char const * * theText = null;

template <typename T> void keyboardFn(unsigned char key, int x, int y)
{
	outln("keyboardFn here key " << key);
   switch (key) {
      case 'q':
         thePersp->setFovy(thePersp->getFovy() * .9);
         outln("fovy " << thePersp->getFovy());
         Drawable<T>::reshapeCb(x,y);
         break;
      case 'w':
          thePersp->setFovy(thePersp->getFovy()* 1.1);
          outln("fovy " << thePersp->getFovy());
          Drawable<T>::reshapeCb(x,y);
          break;
      case 'a':
         thePersp->setAspect(thePersp->getAspect() * .9);
         outln("getAspect " << thePersp->getAspect());
         Drawable<T>::reshapeCb(x,y);
         break;
      case 's':
          thePersp->setAspect(thePersp->getAspect() *1.1);
          outln("getAspect " << thePersp->getAspect());
          Drawable<T>::reshapeCb(x,y);
         break;
      case '1':
          if(p1) {
        	  *p1 -= .01 * (*p1);
        	  outln("pos " << niceVec(p1));
          }
         break;
      case '!':
          if(p1) {
        	  *p1 += .01 * (*p1);
        	  outln("pos " << niceVec(p1));
          }
         break;
      case '2':
          if(p2) {
        	  *p2 -= .01 * (*p2);
        	  outln("pos " << niceVec(p1));
          }
         break;
      case '@':
          if(p2) {
        	  *p2 += .01 * (*p2);
        	  outln("pos " << niceVec(p1));
          }
         break;
      case '3':
           if(p3) {
         	  *p3 -= .01 * (*p3);
        	  outln("pos " << niceVec(p1));
           }
          break;
       case '#':
           if(p3) {
         	  *p3 += .01 * (*p3);
        	  outln("pos " << niceVec(p1));
           }
          break;
       case 'z':
    	   idx++;
    	   if(idx > 5) {
    		   idx = 5;
    	   }
    	   *theText = theVars[idx];
           break;
        case 'x':
        	idx--;
        	if(idx < 0) {
        		idx = 0;
        	}
        	*theText = theVars[idx];
        	break;
        case 'c':
        	*thePoints[idx] -= 1;
            break;
        case 'v':
         	*thePoints[idx] += 1;
         	break;
        case 'p':
         	Drawable<T>::enablePath = !Drawable<T>::enablePath;
         	Drawable<T>::enablePaths();
         	break;
     default:
         break;
   }
}



template int testOglPointAnim<float>::operator()(int argc, char const ** args) const;
template int testOglPointAnim<double>::operator()(int argc, char const ** args) const;
template <typename T> int testOglPointAnim<T>::operator()(int argc, const char** args) const {
	glutInit(&argc, (char**)args);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(750, 750);
	glutInitWindowPosition(100, 100);
	glutCreateWindow(args[0]);
	initPointAnim();
	const T pos[] ={(T).25,(T).25,(T)0};
	Point<T> pt;
	pt.setPos(pos);


	glutDisplayFunc(Drawable<T>::drawCb);
	glutReshapeFunc(reshapeAnim0);
	glutMouseFunc(mousePointAnim<T>);
	glutMainLoop();
	return 0;
}

const char* sevenSamples = "/home/reid/kaggle/accel/frag7.csv";

template int testFrag7Csv<float>::operator()(int argc, char const ** args) const;
template int testFrag7Csv<double>::operator()(int argc, char const ** args) const;
template<typename T> int testFrag7Csv<T>::operator()(int argc, const char** args) const {
	outln( "opening " << sevenSamples);
	map<string, CuMatrix<T>*> results = util<T>::parseCsvDataFile(sevenSamples,",",false, true, false);

	if(!results.size()) {
		outln("no " << sevenSamples << "; exiting");
		return -1;
	}
	typedef typename map<string, CuMatrix<T>*>::iterator iterator;
	iterator it;
	it = results.begin();

	outln("loaded " << sevenSamples);

	CuMatrix<T>& x = *results["x"];
	outln("x " << x);
	CuMatrix<T>means(1,x.n,true,true);
	CuTimer timer;
	timer.start();
	x.featureMeans(means,false);
	outln("mean calc took " << timer.stop()/1000 << "s, means: " << means.syncBuffers());

	AccelEvents<T>* events = AccelEvents<T>::fromMatrix(x);
	//events->syncToNow(1000 );
	outln("events " << events->toString());
	outln("first " << events->first() << ", last " << events->last());

	const T pos[] ={(T)0.0,(T)0,(T)0};
	Bbox<T> bbx(-50.0, 50.0, -50.0, 50.0, -300.0, 300.0);
	Drawable<T>::bbx = &bbx;
	Point<T> pt(pos);
	Drawable<T>::enablePath =false;
	pt.setWithPath( false);
	Perspective persp(80.0, 1.0, 1.5, 70.0);
	LookAt lookAt;
	thePersp = &persp;
	theLookAt = &lookAt;
	thePoints[0] = &theLookAt->eyeX;
	thePoints[1] = &theLookAt->eyeY;
	thePoints[2] = &theLookAt->eyeZ;
	thePoints[3] = &theLookAt->centerX;
	thePoints[4] = &theLookAt->centerY;
	thePoints[5] = &theLookAt->centerZ;

	Drawable<T>::vision = thePersp;
	Drawable<T>::lookAt = theLookAt;
	p1 = (double*)pt.pos;
	p2 = (double*)pt.pos+1;
	p3 = (double*)pt.pos+2;
	outln("made pt " << &pt);
	//pt.setBbx(&bbx);
	pt.setMass(10000);
	outln("after set mass made pt " << niceVec(pt.getGravity()));
	outln("making current");
	pt.makeCurrent();
	Text<T> msg(theVars[idx]);
	theText = msg.getTextAdr();
	T tpos[] = {0,1,15};
	T tnrm[] = {0,1,0};
	msg.set(tpos,tnrm);

	outln("adding events");
	pt.addEvents(events);
	Arrow<T> accArrow;
	pt.setTwin(&accArrow);
	glutInit(&argc, (char**)args);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(750, 750);
	glutInitWindowPosition(100, 100);
	glutCreateWindow(args[0]);
	initPointAnim();

	glutDisplayFunc(Drawable<T>::drawCb);
	glutReshapeFunc(Drawable<T>::reshapeCb);
	glutMouseFunc(mousePointAnim<T>);
	glutKeyboardFunc(keyboardFn<T>);
	glutMainLoop();
	return 0;

	util<CuMatrix<T> >::deletePtrMap(results);

	delete events;

	Drawable<T>::freeDrawList();
	Steppable<T>::freeStepList();
	return 0;
}
#endif // CuMatrix_Enable_Ogl

#include "tests.cc"
