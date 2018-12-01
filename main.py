class LinearRegressionModel:
	def __init__(self, training_data, test_data):
		self.training_data = training_data
		self.test_data = test_data
		self.theta0 = 0
		self.theta1 = 0
		
	def getPrediction(self, x):
		return self.theta0 + x * self.theta1
		
	def derivativeTheta0(self):
		sum = 0
		n = len(self.training_data)
		
		for i in range(n):
			x, y = self.training_data[i]
			sum += (self.getPrediction(x) - y)
		
		deriv = sum / n
		return deriv
		
	def derivativeTheta1(self):
		sum = 0
		n = len(self.training_data)
		
		for i in range(n):
			x, y = self.training_data[i]
			sum += (self.getPrediction(x) - y) * x
		
		deriv = sum / n
		return deriv
		
	def train(self, learning_rate):
		temp0 = self.theta0 - learning_rate * self.derivativeTheta0()
		temp1 = self.theta1 - learning_rate * self.derivativeTheta1()
		
		self.theta0 = temp0
		self.theta1 = temp1
		
	def costFunction(self):
		n = len(self.training_data)
		sum = 0
		
		for i in range(n):
			x, y = self.training_data[i]
			sum += (self.getPrediction(x) - y) ** 2
			
		cost = sum / 2.0 / n
		return cost
		
	def test(self, prec = 10):
		n = len(self.test_data)
		
		for i in range(n):
			x, y = self.test_data[i]
			py = self.getPrediction(x)
			print("Prediction for {0}: {1} (correct={2}, delta={3})".format(x, round(py, prec), y, round(y - py, prec)))
	
	def printFunction(self, prec = 3):
		print("f(x) = {0} + {1} * x".format(round(self.theta0, prec), round(self.theta1, prec)))
	
def main():
	training_data = [(1.0, 100.0), (3.0, 300.0)]
	test_data = [(0.0, 0.0), (1200.0, 120000.0)]
	
	lr = LinearRegressionModel(training_data, test_data)
	print("Initial cost: {0}".format(lr.costFunction()))
	
	for i in range(1, 701):
		lr.train(0.33)
		if i == 1 or i % 25 == 0:
			print("Round {0}: cost={1}".format(i, round(lr.costFunction(), 10)))
		
	lr.test()
	lr.printFunction()
	
if __name__ == "__main__":
	main()
