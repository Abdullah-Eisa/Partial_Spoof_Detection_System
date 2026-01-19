

    # import torch
    # import torch.nn as nn

    def count_params(module, trainable_only=True):
        """Count parameters in a module"""
        if trainable_only:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in module.parameters())


    def print_param_breakdown(module, indent=0):
        """Recursively print parameter count per submodule"""
        prefix = " " * indent
        total = count_params(module)
        print(f"{prefix}{module.__class__.__name__}: {total:,} params")

        for name, child in module.named_children():
            child_params = count_params(child)
            if child_params > 0:
                print(f"{prefix}  ├─ {name}: {child_params:,}")
                print_param_breakdown(child, indent + 4)


    total_params = count_params(PS_Model)
    print(f"\n🔢 Total model parameters: {total_params:,}")

    print("\n🧠 Conformer detailed parameter breakdown:")
    print_param_breakdown(PS_Model.conformer)

    print("\n🧩 FC refinement breakdown:")
    for i, layer in enumerate(PS_Model.fc_refinement):
        print(f"Layer {i} ({layer.__class__.__name__}): {count_params(layer):,}")


