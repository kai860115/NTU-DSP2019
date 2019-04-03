#include "hmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef MODEL_NUM
#define MODEL_NUM 5
#endif

double
test_HMM(HMM *hmm, char *seq)
{
   double delta[MAX_SEQ][MAX_STATE] = {0};
   int seq_num = strlen(seq);
   int state_num = hmm->state_num;
   for (int t = 0; t != seq_num; t++)
   {
      if (t == 0)
      {
         for (int i = 0; i != state_num; i++)
         {
            delta[t][i] = hmm->initial[i] * hmm->observation[seq[t] - 'A'][i];
         }
      }
      else
      {
         for (int j = 0; j != state_num; j++)
         {
            for (int i = 0; i != state_num; i++)
            {
               double tmp = delta[t - 1][i] * hmm->transition[i][j] * hmm->observation[seq[t] - 'A'][j];
               delta[t][j] = (delta[t][j] < tmp) ? tmp : delta[t][j];
            }
         }
      }
   }
   double P = 0;
   for (int i = 0; i != state_num; i++)
      P = P < delta[seq_num - 1][i] ? delta[seq_num - 1][i] : P;
   return P;
}

int main(int argc, char const *argv[])
{
   const char *listname = argv[1];
   const char *testing_data_filename = argv[2];
   const char *result_filename = argv[3];

   HMM models[MODEL_NUM];
   load_models(listname, models, MODEL_NUM);

   FILE *out = open_or_die(result_filename, "w");
   FILE *in = open_or_die(testing_data_filename, "r");

   char seq[MAX_SEQ] = "";
   while (fscanf(in, "%s", seq) > 0)
   {
      double P = 0;
      int maxarg = 0;
      for (int i = 0; i != MODEL_NUM; i++)
      {
         double tmp = test_HMM(&models[i], seq);
         if (tmp > P)
         {
            P = tmp;
            maxarg = i;
         }
      }
      fprintf(out, "%s %e\n", models[maxarg].model_name, P);
   }
   fclose(in);
   fclose(out);

   return 0;
}
