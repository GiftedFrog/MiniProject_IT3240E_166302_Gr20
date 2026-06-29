#include <bits/stdc++.h>
#define endl "\n"
using namespace std;

/*--configuration starts here--*/

const int reportPrecision = 9; // # of digits after the decimal on stage reports; has NO correlation to the answer's precision

const double err = 1e-8; // when the difference between the best and the worst value become smaller than this, the program terminates
const double F = 0.65;
const double CR = 0.7;

const int N = 50;
const int D = 2;

//these are the searching bounds for each parameter
const vector<double> lowerBound = {-1.5, -4.5};
const vector<double> upperBound = {4.5, 1.5};

//the cost function. the parameters are x[0], x[1],... with the bounds defined above
double f(vector<double> &x){
    const double PI = acos(-1.0);

    double term1 = 0.2 * x[0] * x[0];
    double term2 = 0.1 * x[1] * x[1];
    double term3 = cos(2.0 * PI * x[0]) * cos(2.0 * PI * x[1]);
    double term4 = exp(sin(x[0] + x[1]) / 4.0);

    return term1 + term2 - term3 + term4;
}

/*--end of configuration--*/

random_device rd;
mt19937 gen(rd());

uniform_real_distribution<double> random0to1(0, 1);

typedef vector<double> TVector;

int generationID;
vector<TVector> vectorSpace, mutantSpace, trialSpace, selectedSpace;

void PrintVector(TVector v){
    printf("{");
    for (int i = 0; i < D; ++i){
        printf("%.*f", reportPrecision, v[i]);
        if (i < D - 1) cout << ", ";
    }
    printf("}");
}

namespace Initialization{
    void Execute(){
        for (int i = 0; i < N; ++i){
            TVector x;
            for (int j = 0; j < D; ++j){
                x.push_back(0.0);
                x[j] = lowerBound[j] + random0to1(gen) * (upperBound[j] - lowerBound[j]);
            }
            vectorSpace.push_back(x);
        }
    }
}

namespace Mutation{
    uniform_int_distribution<int> randomVectorIndex(0, N - 1);

    vector<int> Pick3RandomVectors(int excludedIndex){
        int x, y, z;

        x = randomVectorIndex(gen);

        y = excludedIndex;
        while ((y == x) || (y == excludedIndex)){
            y = randomVectorIndex(gen);
        }

        z = excludedIndex;
        while ((z == x) || (z == y) || (z == excludedIndex)){
            z = randomVectorIndex(gen);
        }

        vector<int> ret = {x, y, z};
        return ret;
    }

    void Execute(){
        for (int i = 0; i < N; ++i){
            vector<int> r = Pick3RandomVectors(i);
            TVector v;

            for (int j = 0; j < D; ++j){
                v.push_back(0.0);
                double dimValue = vectorSpace[r[0]][j] + F * (vectorSpace[r[1]][j] - vectorSpace[r[2]][j]);
                v[j] = clamp(dimValue, lowerBound[j], upperBound[j]);
            }

            mutantSpace.push_back(v);
        }
    }
}

namespace Crossover{
    uniform_int_distribution<int> randomParameterIndex(0, D - 1);

    void Execute(){
        for (int i = 0; i < N; ++i){
            int jRand = randomParameterIndex(gen);
            TVector u;

            for (int j = 0; j < D; ++j){
                u.push_back(0.0);

                if (j == jRand){
                    u[j] = mutantSpace[i][j];
                    continue;
                }

                double coinFlip = random0to1(gen);
                u[j] = (coinFlip <= CR ? mutantSpace[i][j] : vectorSpace[i][j]);
            }

            trialSpace.push_back(u);
        }
    }
}

namespace Selection{
    void Execute(){
        for (int i = 0; i < N; ++i){
            if (f(trialSpace[i]) <= f(vectorSpace[i])){
                selectedSpace.push_back(trialSpace[i]);
            }
            else{
                selectedSpace.push_back(vectorSpace[i]);
            }
        }
    }
}

namespace GenerationEvaluate{
    string trimOff(double a, double b){
        //just a function that returns the common prefix of a and b in decimal representation
        ostringstream streamA, streamB;

        streamA << fixed << setprecision(reportPrecision) << a;
        streamB << fixed << setprecision(reportPrecision) << b;

        string strA = streamA.str();
        string strB = streamB.str();

        string common = "";
        int minimumLength = min(strA.length(), strB.length());

        for (int i = 0; i < minimumLength; ++i) {
            if (strA[i] == strB[i]) {
                common += strA[i];
            }
            else break;
        }

        if (!common.empty() && (common.back() == '.')) {
            common.pop_back();
        }

        return common;
    }

    void Execute(){
        double bestValue = f(selectedSpace[0]), worstValue = bestValue;
        int bestVectorID = 0, worstVectorID = 0;

        printf("Vector space of Generation #%d:\n\n", generationID);

        for (int i = 0; i < N; ++i){
            printf("    #%d: ", i);
            PrintVector(selectedSpace[i]);
            cout << endl;

            double currentValue = f(selectedSpace[i]);
            if (bestValue > currentValue){
                bestValue = currentValue;
                bestVectorID = i;
            }
            if (worstValue < currentValue){
                worstValue = currentValue;
                worstVectorID = i;
            }
        }

        printf("\nCharacteristics report:\n");

        printf("    f(best) = %.*f, ", reportPrecision, bestValue);
        printf("at #%d: ", bestVectorID);
        PrintVector(selectedSpace[bestVectorID]);
        cout << endl;

        printf("    f(worst) = %.*f, ", reportPrecision, worstValue);
        printf("at #%d: ", worstVectorID);
        PrintVector(selectedSpace[worstVectorID]);
        cout << endl;

        double diff = worstValue - bestValue;
        printf("    diff = %.*f\n\n", reportPrecision, worstValue - bestValue);

        printf("Verdict:\n    ");
        if (diff >= err){
            printf("The values aren't close enough (diff >= err), moving to the next generation...\n");
            printf("\n-----------------------------------------------------------------------------------\n\n");
        }
        else{
            printf("Optimal solution found: f(x*) = %s\n", trimOff(worstValue, bestValue).c_str());
            printf("    The program will now terminate.");
            exit(0);
        }
    }
}

void CleanUp(){
    vectorSpace = selectedSpace;
    mutantSpace.clear();
    trialSpace.clear();
    selectedSpace.clear();
}

void InitReport(){
    cout << "Process started with the following parameters:\n\n";

    printf("    reportPrecision = %d\n", reportPrecision);
    printf("    err = %.*f\n", reportPrecision, err);
    printf("    F = %.*f\n", reportPrecision, F);
    printf("    CR = %.*f\n", reportPrecision, CR);
    printf("    N = %d\n", N);
    printf("    D = %d\n", D);

    printf("    Parameter bounds:\n");
    for (int i = 0; i < D; ++i){
        printf("        x[%d]: [%.*f, %.*f]\n",
               i, reportPrecision, lowerBound[i], reportPrecision, upperBound[i]);
    }

    printf("\nInitialized vector space:\n");
    for (int i = 0; i < N; ++i){
        printf("    #%d: ", i);
        PrintVector(vectorSpace[i]);
        cout << endl;
    }

    cout << "\n-----------------------------------------------------------------------------------\n\n";
}

signed main(){
    freopen("report.txt", "w", stdout);

    Initialization::Execute();
    InitReport();

    while (true){
        ++generationID;
        Mutation::Execute();
        Crossover::Execute();
        Selection::Execute();
        GenerationEvaluate::Execute();
        CleanUp();
    }
}
