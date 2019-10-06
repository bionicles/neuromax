    # def pull_model(self, task_id, task_dict):
    #     print("")
    #     log(f"agent.pull_model {task_id}", color="black_on_white")
    #     outputs, output_roles = [], []
    #     log("we encode the task's id", color="green")
    #     task_id_input = K.Input(self.task_id_spec.shape, batch_size=BATCH_SIZE)
    #     task_code = self.task_sensor(task_id_input)
    #     codes = [task_code]
    #     log("likewise for loss float value", color="green")
    #     loss_input = K.Input((1,), batch_size=BATCH_SIZE)
    #     task_dict.loss_sensor = use_interface(
    #         self, task_id + "loss_sensor", self.loss_spec, self.code_spec)
    #     loss_code = task_dict.loss_sensor(loss_input)
    #     codes.append(loss_code)
    #     inputs = [task_id_input, loss_input]
    #     log(f"we autoencode {task_id} inputs", color="green")
    #     for in_number, in_spec in enumerate(task_dict.inputs):
    #         log(f"input {in_number} needs a sensor and an actuator", color="green")
    #         if in_spec.format is "image":
    #             if self.image_sensor is None:
    #                 self.image_sensor = use_interface(
    #                     self, "image_sensor",
    #                     in_spec=self.image_spec, out_spec=self.code_spec)
    #             if self.image_actuator is None:
    #                 self.image_actuator = use_interface(self, "image_actuator",
    #                                                 self.code_spec, self.image_spec)
    #             sensor = self.image_sensor
    #             actuator = self.image_actuator
    #         else:
    #             sensor = use_interface(self, f"{task_id}_sensor",
    #                                in_spec, self.code_spec,
    #                                in_number=in_number)
    #             actuator = use_interface(self, f"{task_id}_actuator",
    #                                  self.code_spec, in_spec,
    #                                  in_number=in_number)
    #         log("we pass an input to the sensor to get normies & codes",
    #             color="green")
    #         input = K.Input(task_dict.inputs[in_number].shape,
    #                         batch_size=BATCH_SIZE)
    #         normie, input_code = sensor(input)
    #         outputs.append(normie)
    #         output_roles.append(f"normie-{in_number}")
    #         outputs.append(input_code)
    #         output_roles.append(f"code-{in_number}")
    #         codes.append(input_code)
    #         inputs.append(input)
    #         log("now we reconstruct the normie from the code", color="green")
    #         if in_spec.format is "ragged":
    #             placeholder = tf.ones_like(normie)
    #             placeholder = tf.slice(placeholder, [0, 0, 0], [-1, -1, 1])
    #             reconstruction = actuator([input_code, placeholder])
    #         else:
    #             reconstruction = actuator(input_code)
    #         outputs.append(reconstruction)
    #         output_roles.append(f"reconstruction-{in_number}")
    #     log("we make placeholders for agent.graph_model", color="green")
    #     n_placeholders = self.n_in - (2 + len(task_dict.inputs))
    #     if n_placeholders < 1:
    #         log("no placeholders to make...moving on", color="green")
    #     else:
    #         for _ in range(n_placeholders):
    #             prior = tf.random.normal(self.code_spec.shape)
    #             codes.append(prior)
    #     log("we pass codes to GraphModel:", color="green")
    #     graph_outputs = self.graph_model(codes)
    #     log("we make predictions and save them", color="green")
    #     predictions = []
    #     predictor = None
    #     for graph_output_number, graph_output in enumerate(graph_outputs):
    #         graph_out_with_code = L.Concatenate(1)([graph_output, task_code])
    #         log("graph_out_with_code shape", graph_out_with_code.shape,
    #             color="red", debug=1)
    #         if predictor is None:
    #             in_spec = get_spec(
    #                 format="code", shape=graph_out_with_code.shape[1:])
    #             predictor = use_interface(
    #                 self, "predictor", in_spec, self.code_spec)
    #         prediction = predictor(graph_out_with_code)
    #         output_roles.append(f"prediction-{graph_output_number}")
    #         predictions.append(prediction)
    #         outputs.append(prediction)
    #     log("we assemble a world model for the actuators", color="green")
    #     predictions = [
    #         tf.expand_dims(p, 0) if len(p.shape) < 3 else p
    #         for p in predictions]
    #     world_model = tf.concat([*codes, *predictions], 1)
    #     world_model_spec = get_spec(format="code", shape=world_model.shape[1:])
    #     log("we pass the model to actuators to get actions", color="green")
    #     for output_number, out_spec in enumerate(task_dict.outputs):
    #         if out_spec.format is "image":
    #             actuator = self.image_actuator
    #         else:
    #             actuator = use_interface(self, task_id, world_model_spec, out_spec)
    #         if out_spec.format is "ragged":
    #             id, n, index = out_spec.variables[0]
    #             placeholder = tf.ones_like(inputs[n + 2])
    #             placeholder = tf.slice(placeholder, [0, 0, 0], [-1, -1, 1])
    #             action = actuator([world_model, placeholder])
    #         else:
    #             action = actuator(world_model)
    #         outputs.append(action)
    #         output_roles.append(f"action-{output_number}")
    #     log("")
    #     log("we build a model with inputs:", color="green")
    #     [log("input", n, list(i.shape), color="yellow")
    #      for n, i in enumerate(inputs)]
    #     log("")
    #     log("and outputs:", color="green")
    #     self.unpack(output_roles, outputs)
    #     task_model = K.Model(inputs, outputs, name=f"{task_id}_model")
    #     task_dict.output_roles = output_roles
    #     task_dict.model = task_model
    #     show_model(task_model, ".", task_id, "png")
    #     log("")
    #     log(f"SUCCESS! WE BUILT A {task_id.upper()} MODEL!", color="green")
    #     return task_id, task_dict
