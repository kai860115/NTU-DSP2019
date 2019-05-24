#include <stdio.h>
#include <algorithm>
#include <vector>
#include <map>
#include <stack>
#include <string.h>
#include <string>
#include "File.h"
#include "Ngram.h"
#include "Vocab.h"
#include "VocabMap.h"

typedef map<string, vector<string>>::iterator MAP_IT;
typedef vector<string>::iterator V_IT;

// Get P(W2 | W1) -- bigram
double getBigramProb(const char *w1, const char *w2, Vocab &voc, Ngram &lm)
{
    VocabIndex wid1 = voc.getIndex(w1);
    VocabIndex wid2 = voc.getIndex(w2);

    if (wid1 == Vocab_None) //OOV
        wid1 = voc.getIndex(Vocab_Unknown);
    if (wid2 == Vocab_None) //OOV
        wid2 = voc.getIndex(Vocab_Unknown);

    VocabIndex context[] = {wid1, Vocab_None};
    return lm.wordProb(wid2, context);
}

void readMap(File &file, map<string, vector<string>> &m)
{
    char *line;
    while (line = file.getline())
    {
        VocabString words[5000];
        unsigned int nwords = Vocab::parseWords(line, words, 5000);
        string key = words[0];
        for (unsigned i = 1; i != nwords; i++)
        {
            string newWord = words[i];
            m[key].push_back(words[i]);
        }
    }
}

int main(int argc, char const *argv[])
{
    int ngram_order = 0;
    const char *lm_filename;
    const char *map_filename;
    const char *test_filename;

    for (int i = 0; i != 9; i++)
    {
        if (strcmp(argv[i], "-text") == 0)
            test_filename = argv[i + 1];
        else if (strcmp(argv[i], "-map") == 0)
            map_filename = argv[i + 1];
        else if (strcmp(argv[i], "-lm") == 0)
            lm_filename = argv[i + 1];
        else if (strcmp(argv[i], "-order") == 0)
            ngram_order = atoi(argv[i + 1]);
    }

    Vocab voc;
    Ngram lm(voc, ngram_order);

    {
        File lmFile(lm_filename, "r");
        lm.read(lmFile);
        lmFile.close();
    }

    map<string, vector<string>> wordMap;

    {
        File mapFile(map_filename, "r");
        readMap(mapFile, wordMap);
        mapFile.close();
    }

    File testFile(test_filename, "r");
    char *line;

    while (line = testFile.getline())
    {
        VocabString words[100];
        unsigned int nwords = Vocab::parseWords(line, words, 100);
        for (int i = nwords; i != 0; i--)
        {
            words[i] = words[i - 1];
        }
        words[0] = Vocab_SentStart;
        words[nwords + 1] = Vocab_SentEnd;
        nwords += 2;
        vector<LogP> dp[nwords];
        vector<VocabString> word_i[nwords];
        vector<unsigned> backtracking[nwords];

        for (int i = 0; i != nwords; i++)
        {
            if (words[i] == Vocab_SentStart)
            {
                dp[i].push_back(0);
                word_i[i].push_back(Vocab_SentStart);
                backtracking[i].push_back(0);
            }
            else if (words[i] == Vocab_SentEnd)
            {
                unsigned argmax = 0;
                LogP max = LogP_Zero;
                for (unsigned k = 0; k != dp[i - 1].size(); k++)
                {
                    LogP logP = getBigramProb(word_i[i - 1][k], Vocab_SentEnd, voc, lm) + dp[i - 1][k];
                    if (logP > max)
                    {
                        max = logP;
                        argmax = k;
                    }
                }
                dp[i].push_back(max);
                word_i[i].push_back(Vocab_SentEnd);
                backtracking[i].push_back(argmax);
            }
            else
            {
                for (V_IT vit = wordMap[string(words[i])].begin(); vit != wordMap[string(words[i])].end(); vit++)
                {
                    //printf("%d\t%s\n", i, vit->c_str());
                    word_i[i].push_back(vit->c_str());
                    unsigned argmax = 0;
                    LogP max = LogP_Zero;
                    for (unsigned k = 0; k != dp[i - 1].size(); k++)
                    {
                        LogP logP = getBigramProb(word_i[i - 1][k], vit->c_str(), voc, lm) + dp[i - 1][k];
                        //printf("\t\t%s %f\n", word_i[i - 1][k], logP);
                        if (logP > max)
                        {
                            max = logP;
                            argmax = k;
                        }
                    }
                    dp[i].push_back(max);
                    backtracking[i].push_back(argmax);
                }
            }
        }

        stack<VocabString> pred;
        int k = 0;
        for (int i = nwords - 1; i >= 0; i--)
        {
            pred.push(word_i[i][k]);
            k = backtracking[i][k];
        }
        for (int i = 0; i != nwords; i++)
        {
            if (i != nwords - 1)
                printf("%s ", pred.top());
            else
                printf("%s", pred.top());
            pred.pop();
        }
        printf("\n");
    }

    return 0;
}
