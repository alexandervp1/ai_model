#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector <int> numbers = {1,2,3,4,5};
    vector <char> letters = {'a','b','c','d','e'};

    numbers.push_back(69);

    cout << numbers.at(4);

}
