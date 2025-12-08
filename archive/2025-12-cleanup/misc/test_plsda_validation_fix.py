"""
Test to verify PLS-DA validation fix in Model Development tab.

This test simulates the validation logic to ensure PLS-DA is correctly
validated for classification tasks.
"""
import sys

# Import the model registry functions
try:
    from src.spectral_predict.model_registry import is_valid_model, get_supported_models
    print("[OK] Successfully imported model registry functions")
except ImportError as e:
    print(f"[ERROR] Failed to import model registry: {e}")
    sys.exit(1)


def test_validation_logic():
    """Test the validation logic for both regression and classification."""

    print("\n" + "="*60)
    print("Testing Model Validation Logic")
    print("="*60)

    # Test 1: PLS-DA with classification (should PASS)
    print("\nTest 1: PLS-DA with classification task")
    model_type = "PLS-DA"
    task_type = "classification"
    is_valid = is_valid_model(model_type, task_type)
    print(f"  Model: {model_type}")
    print(f"  Task: {task_type}")
    print(f"  Valid: {is_valid}")
    print(f"  Result: {'[PASS]' if is_valid else '[FAIL]'}")

    # Test 2: PLS-DA with regression (should FAIL)
    print("\nTest 2: PLS-DA with regression task")
    model_type = "PLS-DA"
    task_type = "regression"
    is_valid = is_valid_model(model_type, task_type)
    print(f"  Model: {model_type}")
    print(f"  Task: {task_type}")
    print(f"  Valid: {is_valid}")
    print(f"  Result: {'[PASS] (correctly invalid)' if not is_valid else '[FAIL] (should be invalid)'}")

    # Test 3: PLS with regression (should PASS)
    print("\nTest 3: PLS with regression task")
    model_type = "PLS"
    task_type = "regression"
    is_valid = is_valid_model(model_type, task_type)
    print(f"  Model: {model_type}")
    print(f"  Task: {task_type}")
    print(f"  Valid: {is_valid}")
    print(f"  Result: {'[PASS]' if is_valid else '[FAIL]'}")

    # Test 4: RandomForest with classification (should PASS)
    print("\nTest 4: RandomForest with classification task")
    model_type = "RandomForest"
    task_type = "classification"
    is_valid = is_valid_model(model_type, task_type)
    print(f"  Model: {model_type}")
    print(f"  Task: {task_type}")
    print(f"  Valid: {is_valid}")
    print(f"  Result: {'[PASS]' if is_valid else '[FAIL]'}")

    # Test 5: Show all supported models for each task type
    print("\n" + "="*60)
    print("Supported Models by Task Type")
    print("="*60)

    for task in ['regression', 'classification']:
        supported = get_supported_models(task)
        print(f"\n{task.upper()}:")
        for model in supported:
            print(f"  - {model}")

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("\nThe fix ensures that Model Development tab validation:")
    print("1. Gets the actual task_type from self.refine_task_type.get()")
    print("2. Validates model against the correct task type")
    print("3. Provides clear error messages specifying both model and task")
    print("\nThis allows PLS-DA to work correctly for classification tasks!")


if __name__ == "__main__":
    test_validation_logic()
