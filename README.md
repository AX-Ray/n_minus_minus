# N-- 

![C++17](https://img.shields.io/badge/C++-17-blue?style=for-the-badge&logo=c%2B%2B)

---

## Description / Описание

**EN:** A lightweight, self-written C++17 neural network framework designed for educational purposes and experimentation. It features a modular architecture with fully-connected layers, 13 activation functions (including experimental), automatic differentiation via backpropagation, adaptive learning rate, and a complete DQN implementation for reinforcement learning. 

**RU:** Небольшой самописный C++17 фреймворк для нейросетей, предназначенный для образовательных целей и экспериментов. Он отличается модульной архитектурой с полносвязными слоями, 13 функциями активации (включая экспериментальные), автоматическим дифференцированием через обратное распространение, адаптивной скоростью обучения и полноценной реализацией DQN для обучения с подкреплением. 

---

**EN** Simplified example of creating and training a neural network.

**RU** Упрощенный пример создания и обучения нейросети.
```cpp
#include "network.hpp"
#include "activ.hpp"

// EN: Create a network with two hidden layers / RU: Создаём сеть с двумя скрытыми слоями
Network net;
net.add_layer(std::make_unique<Layer>(1, 20, std::make_unique<LSLU>()));       
net.add_layer(std::make_unique<Layer>(20, 20, std::make_unique<LSLU>()));       
net.add_layer(std::make_unique<Layer>(20, 1, std::make_unique<Linear>())); 

// EN: Train on a single sample / RU: Обучение на одном примере
Matrix input(1,1); input(0,0) = 0.5;
Matrix target(1,1); target(0,0) = 0.25;
double loss = net.train(input, target, 0.01);

// EN: Make a prediction / RU: Делаем предсказание
Matrix output = net.predict(input);
net.save("models/model.txt");
```

## Structure / Структура

```
├── build/
├── examples/                    # Examples
│   ├── sin_example              # Approximate sin(x)
│   ├── lunar_landing            # Train DQN on Lunar Lander 
│   └── main.cpp                 # Approximate x²
├── include/                     # Header files
│   ├── activ.hpp                # All activation functions
│   ├── layer.hpp                # Fully‑connected layer
│   ├── matrix.hpp               # Dense matrix class
│   ├── network.hpp              # Neural network container
│   ├── replay_buffer.hpp        # Experience replay for DQN
│   ├── rl_agent.hpp             # DQN agent
│   └── utils.hpp                # Factory for creating activations
├── logs/                        # Training logs
├── models/                      # Saved network weights
├── output/                      # Prediction outputs
├── python/                      # Python utilities
│   ├── activs.py                # Plot all activation functions
│   └── loss.py                  # 3D visualisation of training trajectory (need logs)
├── src/                         # Implementations
│   ├── activ.cpp
│   ├── layer.cpp
│   ├── matrix.cpp
│   ├── network.cpp
│   ├── replay_buffer.cpp
│   └── rl_agent.cpp
└── README.md                    
```

## Build & Run / Сборка и запуск

Prerequisites / Требования

- C++17 compiler (GCC 7+, Clang 5+, MSVC 2017+)

- OpenMP (optional)
```bash
g++ -o build/filename path/filename.cpp src/*.cpp -O3 -march=native -fopenmp
# -fopenmp flag is optional. It can actually decrease perfomance.
```

## Examples / Примеры

**EN** The examples folder contains three programs. main.cpp is the simplest example.

**RU** В папке examples лежат примеры программ. main.cpp самый простой и показательный.

## Warning / Предупреждение
**EN:** This framework is not suitable for serious production tasks, primarily due to the lack of proper GPU acceleration. 

**RU:** Не подойдет для серьезных дел как минимум из за отсутствия внятного взаимодействия с видеокартой. 