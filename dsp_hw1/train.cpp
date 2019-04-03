#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hmm.h"

void train_hmm(HMM *hmm, const char *data_filename, int iterator)
{
   int state_num = hmm->state_num;
   int observ_num = hmm->observ_num;
   char seq[MAX_SEQ];
   for (int n = 0; n != iterator; n++)
   {
      double gamma0_i[MAX_STATE] = {0};
      double gamma_k_i[MAX_OBSERV][MAX_STATE] = {0};
      double gamma_i_observ[MAX_STATE] = {0};
      double gamma_i_tran[MAX_STATE] = {0};
      double epsilon_i_j[MAX_STATE][MAX_STATE] = {0};
      printf("%s train", data_filename);
      printf("  iterator: %d\n", n + 1);
      int N = 0;
      FILE *seq_file = open_or_die(data_filename, "r");

      while (fscanf(seq_file, "%s", seq) > 0)
      {
         N++;
         double alpha[MAX_SEQ][MAX_STATE] = {0.};
         double beta[MAX_SEQ][MAX_STATE] = {0.};
         double gamma[MAX_SEQ][MAX_STATE] = {0.};
         double epsilon[MAX_SEQ][MAX_STATE][MAX_STATE] = {0.};
         int seq_num = strlen(seq);

         for (int t = 0; t != seq_num; t++)
         {
            if (t == 0)
            {
               for (int i = 0; i != state_num; i++)
               {
                  alpha[t][i] = hmm->initial[i] * hmm->observation[seq[t] - 'A'][i];
                  beta[seq_num - t - 1][i] = 1;
               }
            }
            else
            {
               for (int j = 0; j != state_num; j++)
               {
                  for (int i = 0; i != state_num; i++)
                  {
                     alpha[t][j] += alpha[t - 1][i] * hmm->transition[i][j];
                     beta[seq_num - t - 1][j] += hmm->transition[j][i] * hmm->observation[seq[seq_num - t] - 'A'][i] * beta[seq_num - t][i];
                  }
                  alpha[t][j] *= hmm->observation[seq[t] - 'A'][j];
               }
            }
         }

         for (int t = 0; t != seq_num; t++)
         {
            double sigma_gam = 0;
            double sigma_ep = 0;
            for (int i = 0; i != state_num; i++)
            {
               gamma[t][i] = alpha[t][i] * beta[t][i];
               sigma_gam += gamma[t][i];

               if (t < seq_num - 1)
               {
                  for (int j = 0; j != state_num; j++)
                  {
                     epsilon[t][i][j] += alpha[t][i] * hmm->transition[i][j] * hmm->observation[seq[t + 1] - 'A'][j] * beta[t + 1][j];
                     sigma_ep += epsilon[t][i][j];
                  }
               }
            }

            for (int i = 0; i != state_num; i++)
            {
               gamma[t][i] /= sigma_gam;
               if (t < seq_num - 1)
               {
                  for (int j = 0; j != state_num; j++)
                  {
                     epsilon[t][i][j] /= sigma_ep;
                  }
               }
            }
         }

         for (int i = 0; i != state_num; i++)
         {
            gamma0_i[i] += gamma[0][i];
            for (int t = 0; t != seq_num; t++)
            {
               gamma_i_observ[i] += gamma[t][i];
               gamma_k_i[seq[t] - 'A'][i] += gamma[t][i];
               if (t < seq_num - 1)
                  gamma_i_tran[i] += gamma[t][i];
            }
            for (int j = 0; j != state_num; j++)
            {
               for (int t = 0; t != seq_num; t++)
               {
                  if (t < seq_num - 1)
                     epsilon_i_j[i][j] += epsilon[t][i][j];
               }
            }
         }
      }
      fclose(seq_file);
      for (int i = 0; i != state_num; i++)
         hmm->initial[i] = gamma0_i[i] / N;
      for (int i = 0; i != state_num; i++)
         for (int j = 0; j != state_num; j++)
            hmm->transition[i][j] = epsilon_i_j[i][j] / gamma_i_tran[i];

      for (int k = 0; k != MAX_OBSERV; k++)
         for (int i = 0; i != state_num; i++)
            hmm->observation[k][i] = gamma_k_i[k][i] / gamma_i_observ[i];
   }
}

int main(int argc, char const *argv[])
{
   int iter = atoi(argv[1]);
   const char *init_model_name = argv[2];
   const char *data_filename = argv[3];
   const char *out_model_name = argv[4];
   HMM model;
   loadHMM(&model, init_model_name);
   train_hmm(&model, data_filename, iter);
   FILE *out = open_or_die(out_model_name, "w");
   dumpHMM(out, &model);
   fclose(out);

   return 0;
}
