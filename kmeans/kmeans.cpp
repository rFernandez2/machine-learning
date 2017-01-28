#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <climits>
#include <fstream>
#include <vector>

#define MIN(a,b) (((a)<(b))?(a):(b))

#define NUM_LINES 500
#define NUM_CLUSTERS 3
#define EPSILON 0.001

struct point {
    double x, y;
};

typedef struct cluster {
    point centroid;
    std::vector<point> points;
} cluster;

double norm(point p1, point p2) {
    return sqrt(pow(p2.x-p1.x,2) + pow(p2.y - p1.y, 2));
}

point operator+(point p1, point p2) {
    point c = {p1.x+p2.x,p1.y+p2.y};
    return c;
}

point operator/(point p, double x) {
    point c = {p.x / x, p.y / x};
    return c;
}

int stopping_criterion(int *old_a, int *new_a, point *old_c, point *new_c) {
    for(int i = 0; i < NUM_CLUSTERS; i ++) {
        if (fabs(old_c[i].x - new_c[i].x) > EPSILON || fabs(old_c[i].y - new_c[i].y) > EPSILON) {
            std::cout << "old_c: " << old_c[i].x << "\tnew_c: " << new_c[i].x << std::endl;
            return 0;
        }
    }
    for(int i = 0; i < NUM_LINES; i ++) {
        if (old_a[i] != new_a[i]) {
            return 0;
        }
    }
    return 1;
}

//used for k-means ++
double dfunc(point x, point *m, int mlen) {
    double min = INT_MAX;
    double temp;
    for (int i = 0; i < mlen; i ++) {
        temp = norm(x, m[i]);
        min = (temp < min) ? temp : min;
    }
    return min;
}

int main() {
    std::ofstream trace("output.txt");
    std::ofstream trace2;
    trace2.open("kmeans++_distortion.txt", std::ios_base::app);
    point data[NUM_LINES];
    int assignments[NUM_LINES];
    int old_assignments[NUM_LINES];

    std::ifstream read("toydata.txt");
    int lines = NUM_LINES;

    for(int i = 0; i <= lines; i ++) {
        read >> data[i].x;
        read >> data[i].y;
    }

    point centroids[NUM_CLUSTERS];
    point old_centroids[NUM_CLUSTERS];

    cluster clusters[NUM_CLUSTERS];

    int r;
    srand(time(NULL));
    //random initialization of centroids
    for (int i = 0; i < NUM_CLUSTERS; i ++) {
        r = rand() % NUM_LINES;
        centroids[i].x = data[r].x;
        centroids[i].y = data[r].y;
    }

    //k-means++
    /*r = rand() % NUM_LINES;
    centroids[0].x = data[r].x;
    centroids[0].y = data[r].y;
    double total;
    double total_dfunc = 0; //denominator of probability density

    int j;
    for (int i = 1; i < NUM_CLUSTERS; i ++) {
        total_dfunc = 0;
        double r2 = (double) rand() / RAND_MAX; //random number \in[0,1] depends on implementation
        j = 0;

        for (int k = 0; i < NUM_LINES; i ++) { //denominator for prob density
            total_dfunc += pow(dfunc(data[k], centroids, i), 2);
        }

        while(total < r2) {
            total += pow(dfunc(data[j], centroids, i), 2) / total_dfunc;
            j ++;
        }
        centroids[i].x = data[j].x;
        centroids[i].y = data[j].y;
    }*/

    int run_number = 1;
    //carry out algo until local minimum
    do {
        //carry over info
        for (int i = 0; i < NUM_CLUSTERS; i ++) {
            old_centroids[i].x = centroids[i].x;
            old_centroids[i].y = centroids[i].y;
        }
        for (int i = 0; i < NUM_LINES; i ++) {
            old_assignments[i] = assignments[i];
        }

        //assign cluster
        for (int i = 0; i < NUM_LINES; i++) {
            double temp = INT_MAX;
            double vnorm;
            for (int j = 0; j < NUM_CLUSTERS; j++) {
                vnorm = norm(centroids[j], data[i]);
                //std::cout << "vnorm: " << vnorm << std::endl;
                if (vnorm < temp) {
                    //data[i].cluster = j;
                    assignments[i] = j;
                    temp = vnorm;
                }
            }
            clusters[assignments[i]].points.push_back(data[i]);
        }

        //calculate centroid
        for (int i = 0; i < NUM_CLUSTERS; i++) {
            int sz = clusters[i].points.size();

            point average = {0,0};
            for (int j = 0; j < sz; j++) {
                average.x += clusters[i].points[j].x;
                average.y += clusters[i].points[j].y;

            }
            sz = sz ? sz : 1;
            centroids[i].x = average.x / sz;
            centroids[i].y = average.y / sz;
        }

        //calculates distortion function
        double distortion = 0;
        for(int i = 0; i < NUM_CLUSTERS; i ++) {
            int csize = clusters[i].points.size();
            for(int j = 0; j < csize; j ++) {
                distortion += pow(norm(clusters[i].points[j], centroids[i]),2);
            }
        }
        trace2 << run_number << ' ' << distortion << std::endl;
        run_number ++;
        for (int i = 0; i < NUM_CLUSTERS; i ++)
            clusters[i].points.clear(); // start over next time
    } while(!stopping_criterion(old_assignments, assignments, old_centroids, centroids));

    for(int i = 0; i < NUM_LINES; i ++) {
        trace << data[i].x << ' ' << data[i].y << ' ' << assignments[i] << std::endl;
    }

    return 0;
}