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
   int iter = atoi(argv[1]);
   const char *init_model_name = "model_init.txt";
   const char data_filename[5][MAX_LINE] = {"seq_model_01.txt", "seq_model_02.txt", "seq_model_03.txt", "seq_model_04.txt", "seq_model_05.txt"};
   const char out_model_name[5][MAX_LINE] = {"model_01.txt", "model_02.txt", "model_03.txt", "model_04.txt", "model_05.txt"};
   const char *testing_data_filename = "testing_data1.txt";
   const char *acc_filename = "acc_iter.txt";
   HMM model[5];
   for (int i = 0; i != 5; i++)
      loadHMM(&model[i], init_model_name);
   FILE *acc_file = fopen(acc_filename, "w");
   for (int i = 0; i != iter; i++)
   {
      printf("%d ", i);
      for (int j = 0; j != 5; j++)
         train_hmm(&model[j], data_filename[j], 1);
      double correct = 0;
      double total = 0;

      FILE *ans_file = open_or_die("testing_answer.txt", "r");
      FILE *test_file = open_or_die("testing_data1.txt", "r");

      char ans[MAX_LINE] = "";
      char seq[MAX_SEQ] = "";
      while (fscanf(test_file, "%s", seq) > 0 && fscanf(ans_file, "%s", ans) > 0)
      {
         double P = 0;
         int maxarg = 0;
         for (int j = 0; j != 5; j++)
         {
            double tmp = test_HMM(&model[j], seq);
            if (tmp > P)
            {
               P = tmp;
               maxarg = j;
            }
         }
         printf("%s %s\n", out_model_name[maxarg], ans);
         if (strcmp(out_model_name[maxarg], ans) == 0)
         {
            correct++;
         }
         total++;
      }
      fclose(ans_file);
      fclose(test_file);

      fprintf(acc_file, "%f\n", correct / total);
   }
   fclose(acc_file);

   return 0;
}
