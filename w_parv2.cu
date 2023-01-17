#include <stdio.h>
#include <cuda.h>

#define WORDLE_WORD_LIST_MAX_COUNT 13000
#define WORDLE_WORD_LIST_FILE_NAME "valid-wordle-words.txt"

#define WORDLE_ANSWER_LIST_MAX_COUNT 2350
#define WORDLE_ANSWER_LIST_FILE_NAME "wordle-answers-alphabetical.txt"

#define VERBOSE 0
#define PRINT_DEVICE_SETTINGS 0
#define REPETITIONS 1000
#define RANDOM_SEED 100

char *answer;
int valid_word_count;

__host__
void guess(char **guess_to_check, char **result);
__host__
void normalize_memory_col(char **memory, int col_index);
__host__
void normalize_memory_row(char **memory, int row_index);
__host__
void print_memory(char **memory, int **char_count, int **char_count_atleast_flag);
__host__
void init_memory(char **memory);
__host__
int solver_par2(char **valids, char **valids_d);

void guess(char **guess_to_check, char **result){
    char word_correct[6];
    char word_guess[6];
    for (int i = 0; i < 5; i++){
        word_correct[i] = answer[i];
        word_guess[i] = (*guess_to_check)[i];
    }
    word_correct[5] = '\0';
    word_guess[5] = '\0';

    if (VERBOSE){
        printf("-----------------------------------------\n");
        printf("%s\n", word_correct);
        printf("%s\n", word_guess);
    }

    for (int i = 0; i < 5; i++){
        if (word_correct[i] == word_guess[i]){
            (*result)[i] = 'G';
            word_correct[i] = '_';
            word_guess[i] = '_';
        }
        else {
            (*result)[i] = '_';
        }
    }

    int j;
    for (int i = 0; i < 5; i++){ // for word_guess
        for (j = 0; j < 5; j++){ // word_correct
            if (i == j || word_guess[i] == '_' || word_correct[j] == '_')
                continue;

            if (word_guess[i] == word_correct[j]){
                (*result)[i] = 'y';
                word_guess[i] = '_';
                word_correct[j] = '_';
            }
        }   
    }

    (*result)[5] = '\0';
}

__host__
void normalize_memory_col(char **memory, int col_index){
    char key_O = '\0';
    for (int i = 0; i < 5; i++){ // Iterate over each memory column
        if ((*memory)[i * 6 + 1 + col_index] == 'O')
            key_O = (*memory)[i * 6 + 0];
    }
    for (int i = 0; i < 5; i++){ // Other indicators in the same column in other rows must be X, unless their key is the same 
        if ((*memory)[i * 6 + 0] == '.'){
            continue;
        }
        if ((*memory)[i * 6 + 0] == key_O && (*memory)[i * 6 + 1 + col_index] != 'O'){
            (*memory)[i * 6 + 1 + col_index] = 'O';
            normalize_memory_row(memory, i);
        }
        else if ((*memory)[i * 6 + 0] != key_O && (*memory)[i * 6 + 1 + col_index] != 'X'){
            (*memory)[i * 6 + 1 + col_index] = 'X';
            normalize_memory_row(memory, i);
        }
    }
}

__host__
void normalize_memory_row(char **memory, int row_index){
    for (int i = 0; i < 5; i++){ // Iterate over each indicator
        if ((*memory)[row_index * 6 + 1 + i] == 'O')
            normalize_memory_col(memory, i);
    }
}

__host__
void normalize_memory(char **memory){
    for (int i = 0; i < 5; i++)
        normalize_memory_row(memory, i);
}

__host__
void print_memory(char **memory, int **char_count, int **char_count_atleast_flag){
    printf("Memory:\n");
    for (int i = 0; i < 5; i++){
        printf("    %d:  ", i);
        for (int j = 0; j < 6; j++){
            printf("%c", (*memory)[i * 6 + j]);
        }

        if (i == 0){
            printf("    ");
            for (int j = 0; j < 26; j++){
                printf("%c", (char)(j + 97));
            }
        }
        if (i == 1){
            printf("    ");
            for (int j = 0; j < 26; j++){
                if ((*char_count_atleast_flag)[j] == 0 || (*char_count)[j] != 0)
                    printf("%d", (*char_count)[j]);
                else 
                    printf(" ");
            }
        }
        if (i == 2){
            printf("    ");
            for (int j = 0; j < 26; j++){
                if ((*char_count_atleast_flag)[j] == 0)
                    printf("=");
                else 
                    printf(" ");
            }
        }

        printf("\n");
    }
}

__host__
void init_memory(char **memory){
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 6; j++)
            (*memory)[i * 6 + j] = '.';
}

__global__
void compute_score(char *valids_d, char *memory_d, int *char_count_d, int *char_count_atleast_flag_d, int *scores_d, int valid_word_count){
    int word_index = blockIdx.x * blockDim.x + threadIdx.x;

    // Frequencies:
    // https://www3.nd.edu/~busiforc/handouts/cryptography/letterfrequencies.html
    // Frequencies of letters from most to least: eariotnslcudpmhgbfywkvxzjq
    // Python code to generate frequencies: for let in "abcdefghijklmnopqrstuvwxyz": print(26-"eariotnslcudpmhgbfywkvxzjq".index(let), end=", ")
    int frequencies[] = {25, 10, 17, 15, 26, 9, 11, 12, 23, 2, 6, 18, 13, 20, 22, 14, 1, 24, 19, 21, 16, 5, 7, 4, 8, 3};

    if (word_index < valid_word_count){

        int is_valid = 1;
        for (int j = 0; j < 5; j++){ // iterate through the 5 letters
            int letter_as_number = (int)(valids_d[word_index * 5 + j]) - 97;

            // count the number of occurences of the letter
            int occurances = 0;
            for (int k = 0; k < 5; k++){
                int letter_as_number_2 = (int)(valids_d[word_index * 5 + k]) - 97;
                if (letter_as_number == letter_as_number_2)
                    occurances++;
            }

            if ((char_count_atleast_flag_d)[letter_as_number] == 1 && occurances < (char_count_d)[letter_as_number]){
                is_valid = 0;
                break;
            }
            else if ((char_count_atleast_flag_d)[letter_as_number] == 0 && occurances != (char_count_d)[letter_as_number]){
                is_valid = 0;
                break;
            }
        }

        // go through rows of memory 
        for (int j = 0; j < 5; j++){
            char letter_to_test = (memory_d)[j * 6 + 0];
            if (letter_to_test == '.') // row can't be used
                break;

            // key must exist in word (second filter)
            int key_exists = 0;
            for (int k = 0; k < 5; k++){ 
                char letter_in_the_word_to_test = (valids_d)[word_index * 5 + k];
                char indicator = (memory_d)[j * 6 + 1 + k];

                if (key_exists == 0 && letter_in_the_word_to_test == letter_to_test)
                    key_exists = 1;

                if ( // individual X and O indicator testing (third and fourth filter)
                    (letter_in_the_word_to_test == letter_to_test && indicator == 'X') || 
                    (letter_in_the_word_to_test != letter_to_test && indicator == 'O')){
                    is_valid = 0;
                    break;
                }
            }
            if (key_exists == 0)
                is_valid = 0;
        }

        // Compute score
        int score_cumm = 0;
        int letter_done[26];
        for (int j = 0; j < 26; j++)
            letter_done[j] = 0;
        for (int j = 0; j < 5; j++){
            char letter_to_compute_score = (valids_d)[word_index * 5 + j];
            int letter_number = (int)letter_to_compute_score - 97;
            int letter_frequency_score = frequencies[letter_number];

            if (letter_done[letter_number] == 1)
                letter_frequency_score = 0;
            else
                letter_done[letter_number] = 1;
            score_cumm += letter_frequency_score;
        }

        scores_d[word_index] = score_cumm * is_valid;
    }
}

__host__
int get_next_word(char **memory, char **valids_d, int **char_count, int **char_count_atleast_flag){
    // Allocate memory for arrays on device
    char *memory_d;
    int *char_count_d;
    int *char_count_atleast_flag_d;
    int *scores_d;
    int *scores;

    scores = (int*)malloc(valid_word_count * sizeof(int));
    cudaMalloc(&memory_d, 6 * 5 * sizeof(char));
    cudaMalloc(&char_count_d, 26 * sizeof(int));
    cudaMalloc(&char_count_atleast_flag_d, 26 * sizeof(int));
    cudaMalloc(&scores_d, valid_word_count * sizeof(int));

    // Copy to device
    cudaMemcpy(memory_d, *memory, 6 * 5 * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(char_count_d, *char_count, 26 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(char_count_atleast_flag_d, *char_count_atleast_flag, 26 * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 dimBlock(1024, 1, 1);
    dim3 dimGrid(ceil((float)WORDLE_WORD_LIST_MAX_COUNT/1024), 1, 1);

    compute_score<<< dimGrid, dimBlock >>>
    (*valids_d, memory_d, char_count_d, char_count_atleast_flag_d, scores_d, valid_word_count);

    // Get Result
    cudaMemcpy(scores, scores_d, valid_word_count * sizeof(int), cudaMemcpyDeviceToHost);

    // Get Best Word
    int best_word_index = -1;
    int best_word_score = 0;
    for (int i = 0; i < valid_word_count; i++){
        // printf("%d: %d\n", i, scores[i]);
        if (scores[i] > best_word_score){
            best_word_index = i;
            best_word_score = scores[i];
        }
    }

    // Free all memory
    cudaFree(char_count_d);
    cudaFree(char_count_atleast_flag_d);
    cudaFree(memory_d);
    cudaFree(scores);
    free(scores);

    if (VERBOSE)
        printf("Best score %d with index %d\n", best_word_score, best_word_index);

    return best_word_index;
}

__host__
void device_init(char **valids_d, char **valids){
    cudaMalloc(valids_d, valid_word_count * 5 * sizeof(char));
    cudaMemcpy(*valids_d, *valids, valid_word_count * 5 * sizeof(char), cudaMemcpyHostToDevice);
}

__host__
void  device_free(char **valids_d){
    cudaFree(*valids_d);
}

__host__
int solver_par2(char **valids, char **valids_d){
    char *colors = (char*)malloc(6 * sizeof(char));
    char *guess_word = (char*)malloc(6 * sizeof(char));
    int *links = (int*)malloc(5 * sizeof(int)); // used for linking memory rows to guess letters
    int i, j, k;

    int try_count = 0;

    int* char_count = (int*)malloc(26 * sizeof(int));
    int* char_count_atleast_flag = (int*)malloc(26 * sizeof(int));
    for (i = 0; i < 26; i++){
        char_count[i] = 0;
        char_count_atleast_flag[i] = 1;
    }

    char* memory = (char*)malloc(6 * 5 * sizeof(char)); // 5 rows, 6 columns
    init_memory(&memory);

    for (i = 0; i < 10; i++){ // 50 is a killswitch
        try_count++;
        int valid_word_index = 0;

        // Calculate next best guess
        if (i == 0)
            valid_word_index = 10184; // Initial guess will always be SLATE
        else
            valid_word_index = get_next_word(&memory, valids_d, &char_count, &char_count_atleast_flag);

        // Guess the word to get new colors and information
        for (j = 0; j < 6; j++)
            guess_word[j] = (*valids)[valid_word_index * 5 + j];
        guess(&guess_word, &colors);
        if (VERBOSE)
            printf("%s\n", colors);

        // If the colors are all green, exit (solved)
        if (strcmp(colors, "GGGGG") == 0)
            break;

        // Reset all links
        for (j = 0; j < 5; j++)
            links[j] = 0; 

        // Iterate over unempty memory rows (memory rows with Os must be first) 
        // (prioritize existing memory rows before creating new rows)
        for (j = 0; j < 5; j++){
            if (memory[j * 6] == '.')
                continue;

            char memory_row_key = memory[j * 6];

            // Link the next O indicator
            int linked = 0;
            for (k = 0; k < 5; k++){
                if (memory[j * 6 + 1 + k] == 'O' && colors[k] == 'G' && links[k] == 0){
                    links[k] = 1;
                    linked = 1;
                    break;
                }
            }

            if (linked == 0){ // If O indicator does not exist, find a letter to link to a dot (.)
                for (k = 0; k < 5; k++){ // Iterate over each guess word letter
                    if (memory[j * 6 + 1 + k] == 'X')
                        continue;

                    if (memory_row_key == guess_word[k] && links[k] == 0){
                        if (colors[k] == 'G'){
                            memory[j * 6 + 1 + k] = 'O';
                            links[k] = 1;
                        }
                        else if (colors[k] == 'y'){
                            memory[j * 6 + 1 + k] = 'X';
                            links[k] = 1;
                        }
                    }
                }
            }
        }

        // Iterate over guess colors to iterate new memory rows
        for (j = 0; j < 5; j++){ 
            char guess_word_letter = guess_word[j];
            char guess_word_color = colors[j];
            if (links[j] == 0){
                int unused_row_index = -1;
                for (k = 0; k < 5; k++) // Iterate over memory rows
                    if (memory[k * 6] == '.'){
                        unused_row_index = k;
                        break;
                    }

                if (guess_word_color == 'G'){
                    memory[unused_row_index * 6 + 1 + j] = 'O';
                    memory[unused_row_index * 6] = guess_word_letter;
                    char_count[(int)guess_word_letter - 97] += 1;
                }
                else if (guess_word_color == 'y'){
                    memory[unused_row_index * 6 + 1 + j] = 'X';
                    memory[unused_row_index * 6] = guess_word_letter;
                    char_count[(int)guess_word_letter - 97] += 1;
                }
                else {
                    if (char_count_atleast_flag[(int)guess_word_letter - 97] == 1)
                        char_count_atleast_flag[(int)guess_word_letter - 97] = 0;
                }
            }
        }
        normalize_memory(&memory);

        if (VERBOSE)
            print_memory(&memory, &char_count, &char_count_atleast_flag);
    }

    if (VERBOSE){
        printf("-----------------------------------------\n");
        printf("Done! (%d tries)\n", try_count);
    }

    free(colors);
    free(links);
    free(guess_word);
    free(memory);
    free(char_count);

    return try_count;
}

int main(){
    // Printing of device properties
    if (PRINT_DEVICE_SETTINGS){
        int dev_count;
        cudaGetDeviceCount( &dev_count);
        cudaDeviceProp dev_prop;
        for (int i = 0; i < dev_count; i++) {
            cudaGetDeviceProperties( &dev_prop, i);
            printf("DEVICE NUMBER %d:\n", i + 1);
            printf("    name : %s\n", dev_prop.name);
            printf("    clockRate : %d\n", dev_prop.clockRate);
            printf("    maxBlocksPerMultiProcessor : %d\n", dev_prop.maxBlocksPerMultiProcessor);
            printf("    maxThreadsPerBlock : %d\n", dev_prop.maxThreadsPerBlock);
            printf("    maxThreadsPerMultiProcessor : %d\n", dev_prop.maxThreadsPerMultiProcessor);
            printf("    sharedMemPerBlock : %zu\n", dev_prop.sharedMemPerBlock);
            printf("    sharedMemPerBlockOptin : %zu\n", dev_prop.sharedMemPerBlockOptin);
            printf("    sharedMemPerMultiprocessor : %zu\n", dev_prop.sharedMemPerMultiprocessor);
            printf("    warpSize : %d\n", dev_prop.warpSize);
        }
    }

    // Loading of valid words text file
    FILE *textfile;
    char *valids;
    valids = (char*)malloc(WORDLE_WORD_LIST_MAX_COUNT * 5 * sizeof(char));
    valid_word_count = 0;
    textfile = fopen(WORDLE_WORD_LIST_FILE_NAME, "r");
    if(textfile == NULL)
        return 1;
    char tempArray[10];
    while(fgets(tempArray, 10, textfile)){
        valids[valid_word_count * 5 + 0] = tempArray[0];
        valids[valid_word_count * 5 + 1] = tempArray[1];
        valids[valid_word_count * 5 + 2] = tempArray[2];
        valids[valid_word_count * 5 + 3] = tempArray[3];
        valids[valid_word_count * 5 + 4] = tempArray[4];
        valid_word_count++;
    }
    valids[valid_word_count * 5] = '\0';
    fclose(textfile);

    // Loading of valid answers text file
    FILE *textfileAnswers;
    char *answers;
    answers = (char*)malloc(WORDLE_ANSWER_LIST_MAX_COUNT * 5 * sizeof(char));
    int answers_word_count = 0;
    textfileAnswers = fopen(WORDLE_ANSWER_LIST_FILE_NAME, "r");
    if(textfileAnswers == NULL)
        return 1;
    while(fgets(tempArray, 10, textfileAnswers)){
        answers[answers_word_count * 5 + 0] = tempArray[0];
        answers[answers_word_count * 5 + 1] = tempArray[1];
        answers[answers_word_count * 5 + 2] = tempArray[2];
        answers[answers_word_count * 5 + 3] = tempArray[3];
        answers[answers_word_count * 5 + 4] = tempArray[4];
        answers_word_count++;
    }
    answers[answers_word_count * 5] = '\0';
    fclose(textfileAnswers);

    double total_elapsed = 0;
    double total_tries = 0;
    srand(RANDOM_SEED);

    char *valids_d;
    device_init(&valids_d, &valids);

    for (int i = 0; i < REPETITIONS; i++){
        // Prepare answer
        int r = rand();
        int answer_index = r % answers_word_count;
        answer = (char*)malloc(6 * sizeof(char));
        for (int j = 0; j < 5; j++)
            answer[j] = answers[answer_index * 5 + j];
        answer[5] = '\0';
        
        if (VERBOSE)
            printf("Word is: %s\n", answer);

        // Start solver
        // Start recording time
        // https://www.techiedelight.com/find-execution-time-c-program/
        clock_t begin = clock();

        int tries = solver_par2(&valids, &valids_d);

        // Stop recording time and get elapsed time
        clock_t end = clock();
        total_elapsed += ((double)(end - begin) / CLOCKS_PER_SEC);
        total_tries += tries;

        if (VERBOSE == 0)
        printf("%d. Word: %s (%d tries)\n", i + 1,  answer, tries);
    }

    device_free(&valids_d);

    printf("Total elapsed time: %f\n", total_elapsed);
    printf("Average time per word: %f\n", total_elapsed / REPETITIONS);
    printf("Average tries per word: %f\n", total_tries / REPETITIONS);

    // Free memory
    free(valids);
    free(answer);

    return 0;
}