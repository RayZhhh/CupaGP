//
// Created by Derek on 2022/11/17.
//

#ifndef CUDAGPIBC_CLASSIFIER_H
#define CUDAGPIBC_CLASSIFIER_H

#include "eval_gpu.h"
#include <iostream>
#include <vector>

using namespace std;

class BinaryClassifier {
    typedef vector<vector<float>> data_t;
    typedef vector<int> label_t;
    typedef vector<Program> vecP;

public:
    BinaryClassifier(data_t &train_data, label_t &train_label, data_t &valid_data, label_t& valid_label,
                     data_t &test_data, label_t &test_label, int img_h, int img_w) {
        this->train_data = train_data;
        this->train_label = train_label;
        this->valid_data = valid_data;
        this->valid_label = valid_label;
        this->test_data = test_data;
        this->test_label = test_label;
        this->img_h = img_h;
        this->img_w = img_w;
    }

    BinaryClassifier(data_t &train_data, label_t &train_label, data_t &test_data, label_t &test_label,
                     int img_h, int img_w) {
        this->train_data = train_data;
        this->train_label = train_label;
        this->test_data = test_data;
        this->test_label = test_label;
        this->img_h = img_h;
        this->img_w = img_w;
    }

    // GP args
    int population_size = 500;
    const string init_method = "ramped_half_and_half";
    pair<int, int> init_depth = {3, 6};
    int max_program_depth = 8;
    int generations = 50;
    int elist_size = 5;
    int tournament_size = 5;
    float crossover_prob = 0.8;
    float mutation_prob = 0.19;

    // GPU params
    bool use_gpu = true;
    int gpu_id = 0;
    int eval_batch = 10;
    int thread_per_block = 128;

    void init() {
        srand(time(0)); // NOLINT
        population.clear();
        best_program_in_each_gen.clear();
        this->evaluator = new GPUEvaluator(this->train_data, this->train_label,
                                           img_h, img_w, eval_batch, thread_per_block);
    }

    ~BinaryClassifier() {
        delete this->evaluator;
        delete this->valid_evaluator;
        delete this->test_evaluator;
    }

    void train() {
        // population initialization
        population_init();

        // evaluate fitness for the initial population
        evaluator->evaluate_population(population);

        // update and print
        update_generation_properties();
        print_population_properties(0);

        // do iteration
        for (int gen = 1; gen < generations; gen++) {
            vecP new_population;

            // elitism
            vecP temp_pop = population;
            sort(temp_pop.begin(), temp_pop.end(), [](Program &p1, Program &p2) {
                return p1.fitness > p2.fitness;
            });
            for (int i = 0; i < elist_size; i++)
                new_population.emplace_back(temp_pop[i]);

            // generate new population
            for (int i = 0; i < population_size - elist_size; i++) {

                // selection
                auto program = tournament_selection();

                // mutation
                mutation(program);
                new_population.emplace_back(program);
            }

            // update properties of the population
            population = new_population;

            // fitness evaluation
            evaluator->evaluate_population(population);

            // update
            update_generation_properties();
            print_population_properties(gen);
        }
    }

    void run_test() {
        this->valid_evaluator = new GPUEvaluator(valid_data, valid_label, img_h, img_w);
        this->test_evaluator = new GPUEvaluator(test_data, test_label, img_h, img_w);

        // validation
        this->valid_evaluator->evaluate_population(best_program_in_each_gen);
        this->best_test_program.fitness = -1;
        for (auto &i : best_program_in_each_gen) {
            if (i.fitness > best_test_program.fitness) {
                best_test_program = i;
            }
        }

        // test
        this->test_evaluator->evaluate_program(best_test_program);
        cout << endl;
        cout << "[ ======= Run Test ======== ] " << endl;
        cout << "[ Best program in test data ] " << best_test_program.str() << endl;
        cout << "[ Accuracy                  ] " << best_test_program.fitness << endl;
    }

protected:
    // data args
    data_t train_data;
    label_t train_label;
    data_t valid_data;
    label_t valid_label;
    data_t test_data;
    label_t test_label;
    int img_h;
    int img_w;

public:
    // populations
    Program best_program;
    vecP best_program_in_each_gen;
    Program best_test_program;

protected:
    // populations
    vecP population;
    GPUEvaluator *evaluator{};
    GPUEvaluator *valid_evaluator{};
    GPUEvaluator *test_evaluator{};

    void population_init() {
        if (init_method == "ramped_half_and_half") {
            int full_num = population_size / 2;
            int grow_num = population_size - full_num;

            for (int i = 0; i < full_num; i++) {
                int rand_depth = randint_(init_depth.first, init_depth.second);
                population.emplace_back(Program(img_h, img_w, rand_depth, "full"));
            }

            for (int i = 0; i < grow_num; i++) {
                int rand_depth = randint_(init_depth.first, init_depth.second);
                population.emplace_back(Program(img_h, img_w, rand_depth, "grow"));
            }
        }
    }

    Program tournament_selection() {
        int max_index = -1;
        float max_fit = -1;
        for (int i = 0; i < tournament_size; i++) {
            auto rand_index = randint_(0, population_size - 1);
            if (population[rand_index].fitness > max_fit) {
                max_fit = population[rand_index].fitness;
                max_index = rand_index;
            }
        }
        return population[max_index];
    }

    void mutation(Program &program) {
        float prob = random_();
        if (prob < crossover_prob) {
            auto donor = tournament_selection();
            program.crossover(donor);
        } else if (prob < crossover_prob + mutation_prob) {
            if (random_() < 0.5) {
                program.point_mutation();
            } else {
                program.subtree_mutation();
            }
        }

        while (program.depth > max_program_depth) {
            program.hoist_mutation();
        }

        if (program.inner_prefix.size() >= 200) {
            cerr << program.str() << endl;
        }

        assert(program.inner_prefix.size() < 200);
    }

    void update_generation_properties() {
        int best_index = 0;
        float best_fit = -1;

        for (int i = 0; i < population_size; i++) {
            if (population[i].fitness > best_fit) {
                best_fit = population[i].fitness;
                best_index = i;
            }
        }

        best_program = population[best_index];
        best_program_in_each_gen.emplace_back(best_program);
    }

    void print_population_properties(int gen) const {
        cout << "[ Generation   ] " << gen << endl;
        cout << "[ Best Fitness ] " << best_program.fitness << endl;
        cout << "[ Best Program ] " << best_program.str() << endl;
        cout << "[ Length Of It ] " << best_program.inner_prefix.size() << endl;
        cout << "[ Depth Of It  ] " << best_program.depth << endl;
        cout << endl;
    }
};

#endif //CUDAGPIBC_CLASSIFIER_H
