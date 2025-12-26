from typing import Any, Dict
import optuna
from optuna import TrialPruned 


class Optimizer:
    def __init__(
        self,
        EPOCHS_PER_TRIAL: int,
        TOTAL_TRIALS: int,
        OPTIMIZER: str,
        STUDY_NAME="yolo_hyperparameter_optimization",
    ) -> None:
        self.EPOCHS_PER_TRIAL = EPOCHS_PER_TRIAL
        # Número total de trials que o Optuna irá executar
        self.TOTAL_TRIALS = TOTAL_TRIALS
        # Nome do estudo (para salvar e retomar posteriormente)
        self.STUDY_NAME = STUDY_NAME  # "yolo_hyperparameter_optimization"
        # Banco de dados SQLite para armazenar os resultados (opcional)
        self.STUDY_DB_URL = f"sqlite:///{self.STUDY_NAME}.db"
        self.OPTIMIZER = OPTIMIZER

        self.minimize = True

        if self.OPTIMIZER is None or self.OPTIMIZER.lower() == "auto":
            raise ValueError("ESCOLHA UM OTIMIZADOR !")

    def get_hyperparameters(self, trial: optuna.Trial) -> Dict[str, float]:
        """
        Define o espaço de busca dos hiperparâmetros para otimização.

        Esta função especifica os ranges e tipos de hiperparâmetros que o Optuna
        irá explorar durante o processo de otimização. Inclui parâmetros de
        otimizador, loss e data augmentation.

        Args:
            trial (optuna.Trial): Objeto trial do Optuna para sugerir hiperparâmetros

        Returns:
            Dict[str, Any]: Dicionário com os hiperparâmetros sugeridos
        """

        # Parâmetros do Otimizador
        hyperparams = {
            # Taxa de aprendizado inicial (escala logarítmica)
            "lr0": trial.suggest_float("lr0", 1e-4, 1e-1, log=True),
            # Fator da taxa de aprendizado final (lrf = lr0 * lrf no final)
            "lrf": trial.suggest_float("lrf", 1e-4, 1e-1),
            # Momentum do otimizador
            "momentum": trial.suggest_float("momentum", 0.900, 0.9999),
            # Weight decay (regularização L2)
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2),
            # Épocas de warmup
            "warmup_epochs": trial.suggest_float("warmup_epochs", 0.0, 5.0),
            # Warmup momentum
            "warmup_momentum": trial.suggest_float("warmup_momentum", 0.7, 0.99999),
            # Warmup bias lr
            "warmup_bias_lr": trial.suggest_float("warmup_bias_lr", 0.01, 0.2),
            #### Parâmetros de Loss (Ganhos)
            # Peso da loss de bounding box
        }

        return hyperparams

    def objective(self, trial: optuna.Trial) -> float:
        """
        Função objetivo do Optuna que treina o modelo YOLO e retorna a métrica a ser otimizada.

        Esta função é executada para cada trial do Optuna. Ela carrega o modelo YOLO,
        aplica os hiperparâmetros sugeridos, treina o modelo e retorna a soma das
        losses de validação como métrica a ser minimizada.

        Args:
            trial (optuna.Trial): Objeto trial do Optuna

        Returns:
            float: Valor da métrica a ser minimizada (soma das losses de validação)
        """
        
        try:
            # Obter hiperparâmetros sugeridos pelo trial
            hyperparams = self.get_hyperparameters(trial)

            print(f"\n{'='*60}")
            print(f"Iniciando Trial #{trial.number}")
            print(f"{'='*60}")
            print("Hiperparâmetros do trial:")
            for param, value in hyperparams.items():
                print(f"  {param}: {value}")
            print(f"{'='*60}")

            best_val_loss = float("inf")  # Inicialize como infinito
            best_fitness = 0

            def on_fit_epoch_end(trainer):
                """
                Called at the end of each fit epoch (train + validation).
                Capture validation loss for use as Optuna objective.
                """
                epoch = trainer.epoch

                def get_val_loss():
                    nonlocal best_val_loss
                    dfl = trainer.metrics.get("val/dfl_loss")
                    cls = trainer.metrics.get("val/cls_loss")
                    box = trainer.metrics.get("val/box_loss")
                    if any(elem == float("inf") for elem in [dfl, cls, box]):
                        print("infinite loss:")
                        val_loss = float("inf")
                        raise Exception("infinite loss")
                    else:
                        weight_box = hyperparams.get("box", float("inf"))
                        weight_cls = hyperparams.get("cls", float("inf"))
                        weight_dfl = hyperparams.get("dfl", float("inf"))
                        if any(elem == float("inf") for elem in [dfl, cls, box]):
                            print("Val loss weights not found in hyperparameters")
                            raise Exception("INCORRECT WEIGHTS!")
                        val_loss = (
                            weight_box * box + weight_cls * cls + weight_dfl * dfl
                        )

                    # Atualiza melhor loss
                    best_val_loss = min(val_loss, best_val_loss)
                    return val_loss

                # A estrutura abaixo depende da versão da Ultralytics (>=8.0.100). Pode ser adaptada conforme necessário.
                try:
                    # Acessa métricas do val_loader

                    if self.minimize:
                        val_loss = get_val_loss()
                        # Reporta loss para Optuna
                        trial.report(val_loss, step=epoch)
                    else:
                        nonlocal best_fitness
                        current_fitness = trainer.fitness  # 0.1*mAP50 + 0.9*mAP(50,95)
                        epoch = trainer.epoch

                        if current_fitness is not None:
                            best_fitness = max(current_fitness, best_fitness)
                        else:
                            current_fitness = 0

                        trial.report(current_fitness, step=epoch)

                    # Prune se necessário
                    if trial.should_prune():
                        string = "LOSS:" if self.minimize else "FITNESS:"
                        value = best_val_loss if self.minimize else best_fitness
                        print(
                            f"Pruning trial {trial.number} at epoch {epoch} due to poor objective value {string} {value}"
                        )
                        raise TrialPruned()
                except TrialPruned:
                    raise
                except Exception as e:
                    print(f"Erro ao acessar val_loss: {e}")
                    if self.minimize:

                        trial.report(float("inf"), step=epoch)
                    else:
                        trial.report(0, step=epoch)

            def freeze_layer(trainer):
                model = trainer.model
                num_freeze = 15
                print(f"Freezing {num_freeze} layers")
                freeze = [f"model.{x}." for x in range(num_freeze)]  # layers to freeze
                for k, v in model.named_parameters():
                    v.requires_grad = True  # train all layers
                    if any(x in k for x in freeze):
                        # print(f"freezing {k}")
                        v.requires_grad = False
                print(f"{num_freeze} layers are freezed.")

            # model = YOLO(self.PATH_MODELO)
            model = YOLO(self.PATH_MODELO)
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
            model.add_callback("on_train_start", freeze_layer)

            batch = trial.suggest_categorical("batch", [-1, 4, 8, 16])
            # imgsz = trial.suggest_categorical("imgsz", [800, 1000, 1200, 1600])
            
            if not results:
                raise Exception("RESULTADO NULO!")
            # Extrair métricas de validação do último epoch
            # A soma das losses de validação será nossa métrica objetivo
            # print(f"RESULTS DICT: {results}")

            # Pega melhor fitness do treinamento todo
            if self.minimize:

                objective_value = float(best_val_loss)
            else:
                objective_value = float(best_fitness)

            print(f"\nResultados do Trial #{trial.number}:")
            print(f"  fitness: {objective_value:.4f}")
            print(f"  Métrica Objetivo: {objective_value:.4f}")

            # Limpar memória (importante para trials longos)
            del model, results
            gc.collect()
            torch.cuda.empty_cache()
            return objective_value

        except TrialPruned:
            print(f"Trial #{trial.number} foi podado (pruning).")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            raise

        except Exception as e:
            print(f"Erro no Trial #{trial.number}: {str(e)}")
            # Retornar valor alto em caso de erro para que o trial seja descartado
            traceback.print_exc()
            return float("inf")

    def optimization_summary(self, study: optuna.Study) -> dict[Any, Any]:
        """
        Imprime um resumo detalhado dos resultados da otimização.

        Args:
            study (optuna.Study): Estudo Optuna completado
        """

        print("\n" + "=" * 80)
        print("RESUMO DA OTIMIZAÇÃO DE HIPERPARÂMETROS")
        print("=" * 80)

        print(f"Número total de trials executados: {len(study.trials)}")
        print(
            f"Número de trials completados: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}"
        )
        print(
            f"Número de trials com erro: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}"
        )

        if study.best_trial is not None:
            print(f"\nMelhor valor da métrica objetivo: {study.best_trial.value:.6f}")

            print(f"\nMelhores hiperparâmetros encontrados:")
            print("-" * 50)
            melhores_param = study.best_trial.params
            for param, value in melhores_param.items():
                if isinstance(value, float):
                    print(f"  {param:20s}: {value:.6f}")
                else:
                    print(f"  {param:20s}: {value}")

            print(f"\nPara usar estes hiperparâmetros, copie o dicionário abaixo:")
            print("-" * 50)
            print("best_hyperparams = {")
            for param, value in study.best_trial.params.items():
                if isinstance(value, float):
                    print(f"    '{param}': {value:.6f},")
                else:
                    print(f"    '{param}': {value},")
            print("}")
            return melhores_param

        else:
            print("\nNenhum trial foi completado com sucesso!")

        print("=" * 80)

    def funcao_otimizacao(self):
        """
        Função principal que executa a otimização de hiperparâmetros.\n
        Esta função configura o estudo Optuna, executa a otimização e exibe resultados finais.\n
        """

        print("=" * 80)
        print("OTIMIZAÇÃO DE HIPERPARÂMETROS DO YOLOv11 COM OPTUNA")
        print("=" * 80)
        print(f"Dataset: {self.PATH_CONFIG_MODELO}")
        print(f"Modelo: {self.PATH_MODELO}")
        print(f"Épocas por trial: {self.EPOCHS_PER_TRIAL}")
        print(f"Total de trials: {self.TOTAL_TRIALS}")
        print("=" * 80)

        # Verificar se o arquivo do dataset existe
        if not os.path.exists(self.PATH_CONFIG_MODELO):
            print(f"ERRO: Arquivo do dataset não encontrado: {self.PATH_CONFIG_MODELO}")
            print(
                "Por favor, verifique o caminho do arquivo data.yaml na seção de configuração."
            )
            sys.exit(0)

        pruner = optuna.pruners.HyperbandPruner()
        # Criar estudo Optuna
        study = optuna.create_study(
            direction="minimize" if self.minimize else "maximize",
            study_name=self.STUDY_NAME,
            pruner=pruner,
            storage=self.STUDY_DB_URL,  # Salvar no banco SQLite
            load_if_exists=True,  # Retomar estudo existente se houver
        )

        print(f"\nIniciando otimização com {self.TOTAL_TRIALS} trials...")
        # print("Pressione Ctrl+C para interromper a otimização a qualquer momento.\n")
        MAX_JOBS = 1  # NUMERO MAXIMO DE OTIMIZAÇÕES SIMULTANEAS
        # torch.cuda.set_per_process_memory_fraction(1 / MAX_JOBS, device=0)
        try:
            # Executar otimização
            study.optimize(
                self.objective,
                n_trials=self.TOTAL_TRIALS,
                n_jobs=MAX_JOBS,
                timeout=None,
            )

        except KeyboardInterrupt:
            print("\n\nOtimização interrompida pelo usuário.")
            print("Os resultados parciais serão exibidos abaixo.")

        except Exception as e:
            print(f"\nErro durante a otimização: {str(e)}")
            print("Os resultados parciais serão exibidos abaixo.")

        finally:
            # Exibir resumo dos resultados
            dicionario = self.optimization_summary(study)

            # Informações sobre como retomar o estudo
            print(f"\nO estudo foi salvo em: {self.STUDY_DB_URL}")
            print("Para retomar a otimização, execute este script novamente.")
            print(
                "O Optuna automaticamente carregará os trials anteriores e continuará de onde parou."
            )
            nome_db = f"{self.STUDY_NAME}.db"
            if isinstance(study._storage, optuna.storages.RDBStorage):
                study._storage.engine.dispose()
            return dicionario, nome_db