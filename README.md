# Linear Regression Using Gradient Descent
Linear regression can also be performed using a technique called gradient descent, where the coefficients (or weights) of the model are iteratively adjusted to minimize a cost function (usually mean squared error). This method is particularly useful when the number of features is too large for analytical solutions like the normal equation or when the feature matrix is not invertible. The gradient descent algorithm updates the weights by moving in the direction of the negative gradient of the cost function with respect to the weights. The updates occur iteratively until the algorithm converges to a minimum of the cost function. The update rule for each weight is given by :

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msub>
    <mi>&#x3B8;</mi>
    <mi>j</mi>
  </msub>
  <mo>:=</mo>
  <msub>
    <mi>&#x3B8;</mi>
    <mi>j</mi>
  </msub>
  <mo>&#x2212;</mo>
  <mi>&#x3B1;</mi>
  <mfrac>
    <mn>1</mn>
    <mi>m</mi>
  </mfrac>
  <munderover>
    <mo data-mjx-texclass="OP">&#x2211;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>m</mi>
    </mrow>
  </munderover>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">(</mo>
    <msub>
      <mi>h</mi>
      <mrow data-mjx-texclass="ORD">
        <mi>&#x3B8;</mi>
      </mrow>
    </msub>
    <mo stretchy="false">(</mo>
    <msup>
      <mi>x</mi>
      <mrow data-mjx-texclass="ORD">
        <mo stretchy="false">(</mo>
        <mi>i</mi>
        <mo stretchy="false">)</mo>
      </mrow>
    </msup>
    <mo stretchy="false">)</mo>
    <mo>&#x2212;</mo>
    <msup>
      <mi>y</mi>
      <mrow data-mjx-texclass="ORD">
        <mo stretchy="false">(</mo>
        <mi>i</mi>
        <mo stretchy="false">)</mo>
      </mrow>
    </msup>
    <mo data-mjx-texclass="CLOSE">)</mo>
  </mrow>
  <msubsup>
    <mi>x</mi>
    <mi>j</mi>
    <mrow data-mjx-texclass="ORD">
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msubsup>
</math>
Where:

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>&#x3B1;</mi>
</math> is the learning rate,

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>m</mi>
</math> is the number of training examples,

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>h</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>&#x3B8;</mi>
    </mrow>
  </msub>
  <mo stretchy="false">(</mo>
  <msup>
    <mi>x</mi>
    <mrow data-mjx-texclass="ORD">
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo stretchy="false">)</mo>
</math> is the hypothesis function at iteration i .

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi>x</mi>
    <mrow data-mjx-texclass="ORD">
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
</math> is the feature vector of the <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi>i</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>t</mi>
      <mi>h</mi>
    </mrow>
  </msup>
</math> 
 training example

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi>y</mi>
    <mrow data-mjx-texclass="ORD">
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
</math> is the actual target value for the <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi>i</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>t</mi>
      <mi>h</mi>
    </mrow>
  </msup>
</math> 
 training example,

 
 <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msubsup>
    <mi>x</mi>
    <mi>j</mi>
    <mrow data-mjx-texclass="ORD">
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msubsup>
</math> is the value of feature j
 for the <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi>i</mi>
    <mrow data-mjx-texclass="ORD">
      <mi>t</mi>
      <mi>h</mi>
    </mrow>
  </msup>
</math> 
 training example.
 
 The choice of learning rate and the number of iterations are crucial for the convergence and performance of gradient descent. Too small a learning rate may lead to slow convergence, while too large a learning rate may cause overshooting and divergence.
# Practical Implementation
Implementing gradient descent involves initializing the weights, computing the gradient of the cost function, and iteratively updating the weights according to the update rule.