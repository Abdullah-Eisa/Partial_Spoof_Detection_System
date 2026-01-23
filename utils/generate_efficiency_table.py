

# ==========================================================================================



    def calculate_params(model):
        """Calculate total parameters"""
        return sum(p.numel() for p in model.parameters())

    # def calculate_flops(model, input_size=(1, 3, 224, 224), device='cpu'):
    #     """Calculate FLOPs using thop"""
    #     model.to(device)
    #     dummy_input = torch.randn(*input_size).to(device)
    #     flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    #     return flops, params

    # def measure_latency(model, input_size=(1, 3, 224, 224), device='cpu', iterations=100, warmup=10):
    #     """Measure inference latency on CPU/GPU"""
    #     model.to(device)
    #     model.eval()
        
    #     dummy_input = torch.randn(*input_size).to(device)
        
    #     # Warmup
    #     with torch.no_grad():
    #         for _ in range(warmup):
    #             # _ = model(dummy_input)
    #             _ = model(dummy_input,1000,0)
        
    #     # Measure
    #     if device == 'cuda':
    #         torch.cuda.synchronize()
        
    #     start = time.time()
    #     with torch.no_grad():
    #         for _ in range(iterations):
    #             # _ = model(dummy_input)
    #             _ = model(dummy_input,1000,0)

    #     if device == 'cuda':
    #         torch.cuda.synchronize()
        
    #     end = time.time()
    #     latency = ((end - start) / iterations) * 1000  # ms
    #     return latency


    def calculate_flops(model, input_size=(1, 200, 768), device='cpu'):
        """Calculate FLOPs using thop"""
        model.to(device)
        dummy_input = torch.randn(*input_size).to(device)
        # Create dummy lengths and dropout_prob
        dummy_lengths = torch.full((input_size[0],), input_size[1], dtype=torch.int16).to(device)
        dummy_dropout_prob = 0
        flops, params = profile(model, inputs=(dummy_input, dummy_lengths, dummy_dropout_prob), verbose=False)
        return flops, params

    def measure_latency(model, input_size=(1, 200, 768), device='cpu', iterations=100, warmup=10):
        """Measure inference latency on CPU/GPU"""
        model.to(device)
        model.eval()
        
        dummy_input = torch.randn(*input_size).to(device)
        dummy_lengths = torch.full((input_size[0],), input_size[1], dtype=torch.int16).to(device)
        dummy_dropout_prob = 0
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input, dummy_lengths, dummy_dropout_prob)
        
        # Measure
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(dummy_input, dummy_lengths, dummy_dropout_prob)

        if device == 'cuda':
            torch.cuda.synchronize()
        
        end = time.time()
        latency = ((end - start) / iterations) * 1000  # ms
        return latency



    def generate_efficiency_table(model, input_size=(1, 3, 224, 224)):
        """Generate complete efficiency table"""
        print("\n" + "="*70)
        print("EFFICIENCY METRICS TABLE")
        print("="*70)
        
        # CPU metrics
        params = calculate_params(model)
        flops_cpu, _ = calculate_flops(model, input_size, device='cpu')
        latency_cpu = measure_latency(model, input_size, device='cpu')
        
        print(f"\n{'Metric':<20} {'Value':<20} {'Unit'}")
        print("-"*70)
        print(f"{'Parameters':<20} {params/1e6:<20.2f} {'M'}")
        print(f"{'FLOPs (CPU)':<20} {flops_cpu/1e9:<20.2f} {'G'}")
        print(f"{'Latency (CPU)':<20} {latency_cpu:<20.2f} {'ms'}")
        
        # GPU metrics (if available)
        if torch.cuda.is_available():
            flops_gpu, _ = calculate_flops(model, input_size, device='cuda')
            latency_gpu = measure_latency(model, input_size, device='cuda')
            print(f"{'FLOPs (GPU)':<20} {flops_gpu/1e9:<20.2f} {'G'}")
            print(f"{'Latency (GPU)':<20} {latency_gpu:<20.2f} {'ms'}")
        
        print("="*70 + "\n")

    # if __name__ == "__main__":
    #     from torchvision import models
        
    #     # Example with ResNet18
    #     model = models.resnet18(pretrained=False)
    #     generate_efficiency_table(model, input_size=(1, 3, 224, 224))

    generate_efficiency_table(PS_Model, input_size=(1,1000 , 768))


# ==========================================================================================


