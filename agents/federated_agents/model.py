from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, LeakyReLU, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

def create_network(state_space, action_space):
    # map_input: 4, 32x32 binary feature maps
    map_input = Input(shape=(*state_space, 4), name="map_input")

    # macro_obs_input: 8 vals: 
    #   robot location as 1 hot map (20x20)
    #   teammate(s) in range (1)
    #   [teammate_info] = {last goals: (20x20), current_goals(20x20)}
    #   percent complete [1]
    #   [our goals](20x20) * 64

    map_size = state_space[0] * state_space[1]
    input_length = map_size + 1 + map_size + map_size + 1 + map_size # equal to 1602 values for a 20x20 map
    macro_obs_input = Input(shape=(input_length, ), name="macro_observations_input") 

    # First branch is convolutional model to analyse map input
    # CONV2D => LReLu
    x = Conv2D(filters=8, kernel_size=(4,4), strides=(2, 2), name="C1")(map_input)
    x = LeakyReLU()(x)

    # CONV2D => LReLu
    x = Conv2D(filters=16, kernel_size=(3,3), strides=(2, 2),  name="C2")(x)
    x = LeakyReLU()(x)

    # CONV2D => LReLu
    x = Conv2D(filters=16, kernel_size=(2,2), strides=(2, 2),  name="C3")(x)
    x = LeakyReLU()(x)

    # Flattened => FC => LReLu
    x = Flatten()(x)
    x = Dense(32,  name="F1")(x)
    x = LeakyReLU()(x)

    # FC => LReLu
    x = Dense(10,  name="F2")(x)
    x = LeakyReLU()(x)

    x = Model(inputs=map_input, outputs=x)

    # Second branch is a FCL to analyse macro observations
    y = Dense(128, name="F3")(macro_obs_input)
    y = LeakyReLU()(y)
    y = Dropout(0.2)(y)
    y = Model(inputs=macro_obs_input, outputs=y)

    combined = concatenate([x.output, y.output])

    z = Dense(128,  name="F4")(combined)
    z = LeakyReLU()(z)

    z = Dense(128,  name="F5")(z)
    z = LeakyReLU()(z)

    z = Dense(128,  name="F6")(z)
    z = LeakyReLU()(z)

    model_output = Dense(action_space, activation='linear')(z)
    
    inputs = [map_input, macro_obs_input]
    return Model(inputs, model_output, name="CEP")

def model_fn(input_shape, action_space, lr):
    model = create_network(input_shape, action_space)
    optimizer = Adam(learning_rate=lr)

    model.compile(loss='mse', optimizer=optimizer, metrics=["accuracy"])

    return model